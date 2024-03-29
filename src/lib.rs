use std::fmt;
use std::fmt::{Debug, Formatter};
use std::iter::Sum;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

use glam::Vec4Swizzles;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use rasterizer::Rasterizer;

use crate::rasterizer::{DepthTools, FragmentTools};

pub mod clipper;
pub mod presenter;
pub mod rasterizer;

#[derive(Clone, Debug)]
pub struct Buffer<T, const S: usize = 1> {
    data: Vec<[T; S]>,
    width: u32,
    height: u32,
}

impl<T: Copy, const S: usize> Buffer<T, S> {
    pub fn new(width: u32, height: u32) -> Self
    where
        T: Default,
    {
        Self::new_with(width, height, T::default())
    }

    pub fn new_with(width: u32, height: u32, value: T) -> Self {
        assert_ne!(S, 0, "buffer cannot have 0 samples");
        Self {
            data: vec![[value; S]; (width * height) as usize],
            width,
            height,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn resize(&mut self, width: u32, height: u32)
    where
        T: Default,
    {
        self.resize_with(width, height, T::default());
    }

    pub fn resize_with(&mut self, width: u32, height: u32, value: T) {
        self.width = width;
        self.height = height;
        self.data.resize((width * height) as usize, [value; S]);
    }

    pub fn fill(&mut self, value: T) {
        self.data.fill([value; S]);
    }

    fn calculate_index(&self, x: u32, y: u32) -> usize {
        (self.width * y + x) as usize
    }

    pub fn as_slice(&self) -> &[[T; S]] {
        self.data.as_slice()
    }

    pub fn resolve<const N: usize>(&self, dst: &mut Buffer<T, N>)
    where
        T: Sum + Div<f32, Output = T>,
    {
        assert!(
            self.width == dst.width && self.height == dst.height,
            "resolving buffers with different dimmensions not supported",
        );
        for (&src, dst) in self.data.iter().zip(&mut dst.data) {
            let average = src.into_iter().sum::<T>() / S as f32;
            dst.fill(average);
        }
    }

    pub fn as_u8_slice(&self) -> &[u8]
    where
        T: bytemuck::Pod + bytemuck::Zeroable,
    {
        bytemuck::cast_slice(self.as_slice())
    }
}

impl Buffer<Color> {
    pub fn as_f32_slice(&self) -> &[f32] {
        bytemuck::cast_slice(self.as_slice())
    }
}

#[cfg(feature = "image")]
use {image::ImageError, std::path::Path};

#[cfg(feature = "image")]
impl Buffer<Color> {
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ImageError> {
        let image_buffer = image::Rgba32FImage::from_raw(
            self.width(),
            self.height(),
            self.as_f32_slice().to_vec(),
        )
        .unwrap();
        let dynamic_image = image::DynamicImage::ImageRgba32F(image_buffer).flipv();
        dynamic_image.to_rgba8().save(path)
    }
}

impl<T: Copy + Default, const S: usize> Index<[u32; 2]> for Buffer<T, S> {
    type Output = [T; S];

    #[inline]
    fn index(&self, index: [u32; 2]) -> &Self::Output {
        let i = self.calculate_index(index[0], index[1]);
        &self.data[i]
    }
}

impl<T: Copy + Default, const S: usize> IndexMut<[u32; 2]> for Buffer<T, S> {
    #[inline]
    fn index_mut(&mut self, index: [u32; 2]) -> &mut Self::Output {
        let i = self.calculate_index(index[0], index[1]);
        &mut self.data[i]
    }
}

impl<T: Copy + Default> Index<(u32, u32)> for Buffer<T, 1> {
    type Output = T;

    #[inline]
    fn index(&self, index: (u32, u32)) -> &Self::Output {
        &self[[index.0, index.1]][0]
    }
}

impl<T: Copy + Default> IndexMut<(u32, u32)> for Buffer<T, 1> {
    #[inline]
    fn index_mut(&mut self, index: (u32, u32)) -> &mut Self::Output {
        &mut self[[index.0, index.1]][0]
    }
}

impl<T: Copy + Default, const S: usize> Index<[u32; 3]> for Buffer<T, S> {
    type Output = T;

    #[inline]
    fn index(&self, index: [u32; 3]) -> &Self::Output {
        &self[[index[0], index[1]]][index[2] as usize]
    }
}

impl<T: Copy + Default, const S: usize> IndexMut<[u32; 3]> for Buffer<T, S> {
    #[inline]
    fn index_mut(&mut self, index: [u32; 3]) -> &mut Self::Output {
        &mut self[[index[0], index[1]]][index[2] as usize]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[rustfmt::skip]
impl Color {
    pub const TRANSPARENT: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };
    pub const BLACK: Self       = Self { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const WHITE: Self       = Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const RED: Self         = Self { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const GREEN: Self       = Self { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const BLUE: Self        = Self { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };
}

impl Color {
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn to_vec4(&self) -> glam::Vec4 {
        glam::vec4(self.r, self.g, self.b, self.a)
    }

    pub fn to_array(&self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

impl Add for Color {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
            a: self.a + rhs.a,
        }
    }
}

impl Div<f32> for Color {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self {
            r: self.r / rhs,
            g: self.g / rhs,
            b: self.b / rhs,
            a: self.a / rhs,
        }
    }
}

impl Sum for Color {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Color::new(0.0, 0.0, 0.0, 0.0), |color1, color2| {
            color1 + color2
        })
    }
}

impl From<glam::Vec4> for Color {
    fn from(vec: glam::Vec4) -> Self {
        Self {
            r: vec.x,
            g: vec.y,
            b: vec.z,
            a: vec.w,
        }
    }
}

impl From<Color> for glam::Vec4 {
    fn from(color: Color) -> Self {
        color.to_vec4()
    }
}

impl From<[f32; 4]> for Color {
    fn from(array: [f32; 4]) -> Self {
        Self {
            r: array[0],
            g: array[1],
            b: array[2],
            a: array[3],
        }
    }
}

impl From<Color> for [f32; 4] {
    fn from(color: Color) -> Self {
        color.to_array()
    }
}

#[derive(Copy, Clone)]
pub struct DepthState {
    pub depth_function: fn(&f32, &f32) -> bool,
    pub write_depth: bool,
}

impl Debug for DepthState {
    fn fmt<'a>(&'a self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DepthState")
            .field(&self.depth_function as &fn(&'a f32, &'a f32) -> bool)
            .field(&self.write_depth)
            .finish()
    }
}

pub mod blend_function {
    use crate::Color;

    #[inline]
    pub fn replace(src: &Color, _: &Color) -> Color {
        *src
    }

    #[inline]
    pub fn alpha_blend(src: &Color, dst: &Color) -> Color {
        let comp_a = src.a + (1.0 - src.a) * dst.a;
        Color {
            r: (src.r * src.a + (1.0 - src.a) * dst.r * dst.a) / comp_a,
            g: (src.g * src.a + (1.0 - src.a) * dst.g * dst.a) / comp_a,
            b: (src.b * src.a + (1.0 - src.a) * dst.b * dst.a) / comp_a,
            a: comp_a,
        }
    }

    #[inline]
    pub fn precomputed_alpha_blend(src: &Color, dst: &Color) -> Color {
        Color {
            r: src.r + (1.0 - src.a) * dst.r,
            g: src.g + (1.0 - src.a) * dst.g,
            b: src.b + (1.0 - src.a) * dst.b,
            a: src.a + (1.0 - src.a) * dst.a,
        }
    }
}

pub trait Multisampler<const S: usize> {
    fn x_offsets(&self, x: u32, y: u32) -> [f32; S];
    fn y_offsets(&self) -> &[f32; S];
}

#[derive(Copy, Clone, Debug)]
pub struct StaticMultisampler<const S: usize> {
    pub x_offsets: [f32; S],
    pub y_offsets: [f32; S],
}

impl<const S: usize> Multisampler<S> for StaticMultisampler<S> {
    fn x_offsets(&self, _: u32, _: u32) -> [f32; S] {
        self.x_offsets
    }

    fn y_offsets(&self) -> &[f32; S] {
        &self.y_offsets
    }
}

impl<const S: usize> StaticMultisampler<S> {
    pub fn from_sample_pattern(pattern: &[[u8; S]; S]) -> Self {
        let mut x_offsets = [0.0; S];
        let mut y_offsets = [0.0; S];
        let mut count = 0;
        let step = 1.0 / S as f32;

        for x in 0..S {
            for y in 0..S {
                if pattern[x][y] == 1 {
                    if count < S {
                        x_offsets[count] = x as f32 * step + step * 0.5;
                        y_offsets[count] = y as f32 * step + step * 0.5;
                        count += 1;
                    } else {
                        panic!("number of samples in sample pattern not equal to pattern size");
                    }
                }
            }
        }

        assert_eq!(
            count, S,
            "number of samples in sample pattern not equal to pattern size",
        );

        Self {
            x_offsets,
            y_offsets,
        }
    }
}

impl StaticMultisampler<1> {
    pub fn single_sample() -> Self {
        Self::from_sample_pattern(&[[1]])
    }
}

impl StaticMultisampler<4> {
    pub fn x4() -> Self {
        #[rustfmt::skip]
        let pattern = [
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ];
        Self::from_sample_pattern(&pattern)
    }
}

impl StaticMultisampler<8> {
    pub fn x8() -> Self {
        Self::from_sample_pattern(&[
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
        ])
    }
}

impl StaticMultisampler<16> {
    pub fn x16() -> Self {
        Self::from_sample_pattern(&[
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        ])
    }
}

#[derive(Copy, Clone, Debug)]
pub struct StochasticMultisampler<const S: usize> {
    offsets: [f32; S],
}

impl<const S: usize> StochasticMultisampler<S> {
    pub fn new() -> Self {
        let step = 1.0 / S as f32;
        let mut offsets = [0.0; S];

        for i in 0..S {
            offsets[i] = i as f32 * step + step * 0.5;
        }

        Self { offsets }
    }
}

impl<const S: usize> Multisampler<S> for StochasticMultisampler<S> {
    fn x_offsets(&self, x: u32, y: u32) -> [f32; S] {
        let mut rng = SmallRng::seed_from_u64(((x as u64) << 32) | y as u64);
        let mut x_offsets = self.offsets;
        x_offsets.shuffle(&mut rng);
        x_offsets
    }

    fn y_offsets(&self) -> &[f32; S] {
        &self.offsets
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PipelineDescriptor<VS, SA, C, R, FS, B, MS> {
    pub vertex_shader: VS,
    pub shape_assembler: SA,
    pub clipper: C,
    pub rasterizer: R,
    pub depth_state: Option<DepthState>,
    pub fragment_shader: FS,
    pub blend_function: B,
    pub multisampler: MS,
}

#[derive(Copy, Clone, Debug)]
pub struct Pipeline<VS, SA, C, R, FS, B, MS, const A: usize, const N: usize, const S: usize> {
    vertex_shader: VS,
    shape_assembler: SA,
    clipper: C,
    rasterizer: R,
    depth_state: Option<DepthState>,
    fragment_shader: FS,
    blend_function: B,
    multisampler: MS,
}

impl<VS, SA, C, R, FS, B, MS, const A: usize, const N: usize, const S: usize>
    Pipeline<VS, SA, C, R, FS, B, MS, A, N, S>
where
    SA: ShapeAssembler<u32, N>,
    C: FnMut([UnshadedFragment<A>; N]) -> Vec<[UnshadedFragment<A>; N]>,
    R: Rasterizer<N, A, S>,
    MS: Multisampler<S> + Sync,
{
    pub fn new(descriptor: PipelineDescriptor<VS, SA, C, R, FS, B, MS>) -> Self {
        Self {
            vertex_shader: descriptor.vertex_shader,
            shape_assembler: descriptor.shape_assembler,
            clipper: descriptor.clipper,
            rasterizer: descriptor.rasterizer,
            depth_state: descriptor.depth_state,
            fragment_shader: descriptor.fragment_shader,
            blend_function: descriptor.blend_function,
            multisampler: descriptor.multisampler,
        }
    }

    pub fn draw_indexed<VI, U, T>(
        &mut self,
        vertex_buffer: &[VI],
        index_buffer: &[u32],
        uniforms: U,
        render_buffer: &mut Buffer<T, S>,
        depth_buffer: Option<&mut Buffer<f32, S>>,
    ) where
        VI: Copy,
        U: Copy + Send,
        T: Copy + Default + Send,
        VS: Fn(VI, U) -> UnshadedFragment<A>,
        FS: Fn(UnshadedFragment<A>, U) -> T + Sync,
        B: Fn(&T, &T) -> T + Sync,
    {
        assert!(
            !(self.depth_state.is_some() && depth_buffer.is_none()),
            "cannot draw using pipeline that requires depth buffer without a depth buffer",
        );
        let width = render_buffer.width();
        let height = render_buffer.height();

        let perspective_divide = |shape: [UnshadedFragment<A>; N]| {
            shape.map(|vertex| {
                let inv_w = 1.0 / vertex.position.w;
                let position: glam::Vec4 = (vertex.position.xyz() * inv_w, inv_w).into();
                let xy = (position.xy() + 1.0) * glam::vec2(width as f32, height as f32) / 2.0;
                UnshadedFragment {
                    position: (xy, position.zw()).into(),
                    attributes: vertex.attributes.map(|attribute| attribute * inv_w),
                }
            })
        };

        let mut fragment_tools = FragmentTools {
            uniforms,
            fragment_shader: &self.fragment_shader,
            blend_function: &self.blend_function,
            render_buffer,
        };

        let mut depth_tools = if let Some(depth_state) = &self.depth_state {
            let depth_buffer = depth_buffer.unwrap();
            assert!(
                width == depth_buffer.width() && height == depth_buffer.height(),
                "render_buffer and depth_buffer size mismatch",
            );
            Some(DepthTools {
                depth_state,
                depth_buffer,
            })
        } else {
            None
        };

        self.rasterizer.set_framebuffer_size(width, height);
        let unshaded_fragments = vertex_buffer
            .into_iter()
            .map(|&vertex| (self.vertex_shader)(vertex, uniforms))
            .collect::<Vec<_>>();
        index_buffer
            .into_iter()
            .map(u32::to_owned)
            .assemble_shapes(self.shape_assembler)
            .map(|shape| shape.map(|i| unshaded_fragments[i as usize]))
            .flat_map(|shape| (self.clipper)(shape).into_iter())
            .map(perspective_divide)
            .for_each(|shape| {
                self.rasterizer.rasterize(
                    shape,
                    &mut fragment_tools,
                    &mut depth_tools,
                    &self.multisampler,
                );
            });
    }

    pub fn draw<VI, U, T>(
        &mut self,
        vertex_buffer: &[VI],
        uniforms: U,
        render_buffer: &mut Buffer<T, S>,
        depth_buffer: Option<&mut Buffer<f32, S>>,
    ) where
        VI: Copy,
        U: Copy + Send,
        T: Copy + Default + Send,
        VS: Fn(VI, U) -> UnshadedFragment<A>,
        FS: Fn(UnshadedFragment<A>, U) -> T + Sync,
        B: Fn(&T, &T) -> T + Sync,
    {
        let index_buffer = (0..vertex_buffer.len() as u32).collect::<Vec<_>>();
        self.draw_indexed(
            vertex_buffer,
            &index_buffer,
            uniforms,
            render_buffer,
            depth_buffer,
        );
    }
}

#[derive(Copy, Clone, Debug)]
pub struct UnshadedFragment<const A: usize = 0> {
    pub position: glam::Vec4,
    pub attributes: [f32; A],
}

impl<const A: usize> Add for UnshadedFragment<A> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut attributes = [0.0; A];
        for i in 0..A {
            attributes[i] = self.attributes[i] + rhs.attributes[i];
        }
        Self {
            position: self.position + rhs.position,
            attributes,
        }
    }
}

impl<const A: usize> Sub for UnshadedFragment<A> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut attributes = [0.0; A];
        for i in 0..A {
            attributes[i] = self.attributes[i] - rhs.attributes[i];
        }
        Self {
            position: self.position - rhs.position,
            attributes,
        }
    }
}

impl<const A: usize> Mul<f32> for UnshadedFragment<A> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            position: self.position * rhs,
            attributes: self.attributes.map(|attr| attr * rhs),
        }
    }
}

pub trait ShapeAssembler<V: Copy, const N: usize>: Copy {
    fn init(&mut self, iter: &mut impl Iterator<Item = V>);
    fn next(&mut self, iter: &mut impl Iterator<Item = V>) -> Option<[V; N]>;
    fn size_hint(&self, _: &impl Iterator<Item = V>) -> (usize, Option<usize>) {
        (0, None)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct StripShapeAssembler<V: Copy, const N: usize> {
    shape: Option<[V; N]>,
}

impl<V: Copy, const N: usize> StripShapeAssembler<V, N> {
    pub fn new() -> Self {
        Self { shape: None }
    }
}

impl<V: Copy, const N: usize> ShapeAssembler<V, N> for StripShapeAssembler<V, N> {
    fn init(&mut self, iter: &mut impl Iterator<Item = V>) {
        let mut shape = Vec::with_capacity(N);
        for _ in 0..N {
            if let Some(vertex) = iter.next() {
                shape.push(vertex);
            }
        }
        self.shape = shape.try_into().ok();
    }

    fn next(&mut self, iter: &mut impl Iterator<Item = V>) -> Option<[V; N]> {
        if let Some(ref mut shape) = self.shape {
            let res = *shape;
            if let Some(vertex) = iter.next() {
                for i in 1..N {
                    shape[i - 1] = shape[i];
                }
                shape[N - 1] = vertex;
            } else {
                self.shape = None;
            }
            Some(res)
        } else {
            None
        }
    }

    fn size_hint(&self, iter: &impl Iterator<Item = V>) -> (usize, Option<usize>) {
        let (lower, upper) = iter.size_hint();
        (lower, upper.map(|bound| bound + 1))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ListShapeAssembler<V: Copy, const N: usize> {
    shape: Option<[V; N]>,
}

impl<V: Copy, const N: usize> ListShapeAssembler<V, N> {
    pub fn new() -> Self {
        Self { shape: None }
    }
}

impl<V: Copy, const N: usize> ShapeAssembler<V, N> for ListShapeAssembler<V, N> {
    fn init(&mut self, iter: &mut impl Iterator<Item = V>) {
        let mut shape = Vec::with_capacity(N);
        for _ in 0..N {
            if let Some(vertex) = iter.next() {
                shape.push(vertex);
            }
        }
        self.shape = shape.try_into().ok();
    }

    fn next(&mut self, iter: &mut impl Iterator<Item = V>) -> Option<[V; N]> {
        if let Some(ref mut shape) = self.shape {
            let res = *shape;
            for vertex in shape {
                if let Some(next_vertex) = iter.next() {
                    *vertex = next_vertex;
                } else {
                    self.shape = None;
                    break;
                }
            }
            Some(res)
        } else {
            None
        }
    }

    fn size_hint(&self, iter: &impl Iterator<Item = V>) -> (usize, Option<usize>) {
        let (lower, upper) = iter.size_hint();
        (lower, upper.map(|bound| bound / 2 + 1))
    }
}

#[derive(Debug)]
struct ShapeAssemblerIterator<I, SA, V, const N: usize>
where
    I: Iterator<Item = V>,
    SA: ShapeAssembler<V, N>,
    V: Copy,
{
    iter: I,
    shape_assembler: SA,
}

impl<I, SA, V, const N: usize> ShapeAssemblerIterator<I, SA, V, N>
where
    I: Iterator<Item = V>,
    SA: ShapeAssembler<V, N>,
    V: Copy,
{
    fn new(mut iter: I, mut shape_assembler: SA) -> Self {
        shape_assembler.init(&mut iter);
        Self {
            iter,
            shape_assembler,
        }
    }
}

impl<I, SA, V, const N: usize> Iterator for ShapeAssemblerIterator<I, SA, V, N>
where
    I: Iterator<Item = V>,
    SA: ShapeAssembler<V, N>,
    V: Copy,
{
    type Item = [V; N];

    fn next(&mut self) -> Option<Self::Item> {
        self.shape_assembler.next(&mut self.iter)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.shape_assembler.size_hint(&self.iter)
    }
}

trait ShapeAssemblerIteratorTrait<SA, V, const N: usize>: Iterator<Item = V> + Sized
where
    SA: ShapeAssembler<V, N>,
    V: Copy,
{
    fn assemble_shapes(self, shape_assembler: SA) -> ShapeAssemblerIterator<Self, SA, V, N> {
        ShapeAssemblerIterator::new(self, shape_assembler)
    }
}

impl<I, SA, const N: usize> ShapeAssemblerIteratorTrait<SA, I::Item, N> for I
where
    I: Iterator,
    I::Item: Copy,
    SA: ShapeAssembler<I::Item, N>,
{
}
