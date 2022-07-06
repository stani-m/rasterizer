use std::ops::{Index, IndexMut};

use glam::Vec4Swizzles;

use rasterizer::Rasterizer;

pub mod clipper;
pub mod presenter;
pub mod rasterizer;

#[derive(Clone, Debug)]
pub struct Buffer<T: Copy + Default, const S: usize> {
    data: Vec<[T; S]>,
    width: u32,
    height: u32,
}

impl<T: Copy + Default, const S: usize> Buffer<T, S> {
    pub fn new(width: u32, height: u32) -> Self {
        let size = width * height;
        Self {
            data: vec![[T::default(); S]; size as usize],
            width,
            height,
        }
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.data
            .resize((width * height) as usize, [T::default(); S]);
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
}

impl<T: Copy + Default + bytemuck::Pod + bytemuck::Zeroable, const S: usize> Buffer<T, S> {
    pub fn as_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.data.as_slice())
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

#[derive(Copy, Clone, Debug)]
pub struct DepthState<DF> {
    pub depth_function: DF,
    pub write_depth: bool,
}

pub mod depth_function {
    #[inline]
    pub fn less_or_equal(src: f32, dst: f32) -> bool {
        dst >= src
    }

    #[inline]
    pub fn less(src: f32, dst: f32) -> bool {
        dst > src
    }

    #[inline]
    pub fn greater_or_equal(src: f32, dst: f32) -> bool {
        dst <= src
    }

    #[inline]
    pub fn greater(src: f32, dst: f32) -> bool {
        dst < src
    }
    #[inline]

    pub fn always_pass(_: f32, _: f32) -> bool {
        true
    }

    #[inline]
    pub fn always_fail(_: f32, _: f32) -> bool {
        false
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

#[derive(Copy, Clone, Debug)]
pub struct PipelineDescriptor<VS, SA, C, R, DF, FS, B> {
    pub vertex_shader: VS,
    pub shape_assembler: SA,
    pub clipper: C,
    pub rasterizer: R,
    pub depth_state: Option<DepthState<DF>>,
    pub fragment_shader: FS,
    pub blend_function: B,
}

#[derive(Copy, Clone, Debug)]
pub struct Pipeline<VS, SA, C, R, DF, FS, B, const A: usize, const V: usize> {
    vertex_shader: VS,
    shape_assembler: SA,
    clipper: C,
    rasterizer: R,
    depth_state: Option<DepthState<DF>>,
    fragment_shader: FS,
    blend_function: B,
}

impl<VS, SA, C, R, DF, FS, B, const A: usize, const V: usize>
    Pipeline<VS, SA, C, R, DF, FS, B, A, V>
where
    SA: ShapeAssembler<u32, V>,
    C: FnMut([FragmentInput<A>; V]) -> Vec<[FragmentInput<A>; V]>,
    R: Rasterizer<A, V>,
    DF: Fn(f32, f32) -> bool,
{
    pub fn new(descriptor: PipelineDescriptor<VS, SA, C, R, DF, FS, B>) -> Self {
        Self {
            vertex_shader: descriptor.vertex_shader,
            shape_assembler: descriptor.shape_assembler,
            clipper: descriptor.clipper,
            rasterizer: descriptor.rasterizer,
            depth_state: descriptor.depth_state,
            fragment_shader: descriptor.fragment_shader,
            blend_function: descriptor.blend_function,
        }
    }

    pub fn draw_indexed<VI, U, T>(
        &mut self,
        vertex_buffer: &[VI],
        index_buffer: &[u32],
        uniforms: U,
        render_buffer: &mut Buffer<T, 1>,
        mut depth_buffer: Option<&mut Buffer<f32, 1>>,
    ) where
        VI: Copy,
        U: Copy,
        T: Copy + Default,
        VS: Fn(VI, U) -> FragmentInput<A>,
        FS: Fn(FragmentInput<A>, U) -> T,
        B: Fn(&T, &T) -> T,
    {
        if self.depth_state.is_some() && depth_buffer.is_none() {
            panic!("Attempting to draw using pipeline that requires depth buffer without a depth buffer");
        }
        let width = render_buffer.width();
        let height = render_buffer.height();

        let perspective_divide = |shape: [FragmentInput<A>; V]| {
            shape.map(|vertex| {
                let inv_w = 1.0 / vertex.position.w;
                let position: glam::Vec4 = (vertex.position.xyz() * inv_w, inv_w).into();
                let xy = (position.xy() + 1.0) * glam::vec2(width as f32, height as f32) / 2.0;
                FragmentInput {
                    position: (xy, position.zw()).into(),
                    attributes: vertex.attributes.map(|attribute| attribute * inv_w),
                }
            })
        };

        let mut rasterizer_action = |fragment_input: FragmentInput<A>| {
            let [x, y] = fragment_input.position.xy().as_uvec2().to_array();

            if let Some(depth_state) = &self.depth_state {
                let depth_buffer = depth_buffer.as_mut().unwrap();
                let src_depth = fragment_input.position.z;
                let dst_depth = depth_buffer[[x, y, 0]];
                let depth_test_passed = (depth_state.depth_function)(src_depth, dst_depth);

                if depth_test_passed {
                    let src_color = (self.fragment_shader)(fragment_input, uniforms);
                    let dst_color = render_buffer[[x, y, 0]];
                    render_buffer[[x, y, 0]] = (self.blend_function)(&src_color, &dst_color);

                    if depth_state.write_depth {
                        depth_buffer[[x, y, 0]] = src_depth;
                    }
                }
            } else {
                let src_color = (self.fragment_shader)(fragment_input, uniforms);
                let dst_color = render_buffer[[x, y, 0]];
                render_buffer[[x, y, 0]] = (self.blend_function)(&src_color, &dst_color);
            }
        };

        self.rasterizer.set_screen_size(width, height);
        let shaded_vertices = vertex_buffer
            .into_iter()
            .map(|&vertex| (self.vertex_shader)(vertex, uniforms))
            .collect::<Vec<_>>();
        index_buffer
            .into_iter()
            .map(u32::to_owned)
            .assemble_shapes(self.shape_assembler)
            .map(|shape| shape.map(|i| shaded_vertices[i as usize]))
            .map(|shape| (self.clipper)(shape).into_iter())
            .flatten()
            .map(perspective_divide)
            .for_each(|shape| self.rasterizer.rasterize(shape, &mut rasterizer_action));
    }

    pub fn draw<VI, U, T>(
        &mut self,
        vertex_buffer: &[VI],
        unforms: U,
        render_buffer: &mut Buffer<T, 1>,
        depth_buffer: Option<&mut Buffer<f32, 1>>,
    ) where
        VI: Copy,
        U: Copy,
        T: Copy + Default,
        VS: Fn(VI, U) -> FragmentInput<A>,
        FS: Fn(FragmentInput<A>, U) -> T,
        B: Fn(&T, &T) -> T,
    {
        let index_buffer = (0..vertex_buffer.len() as u32).collect::<Vec<_>>();
        self.draw_indexed(
            vertex_buffer,
            &index_buffer,
            unforms,
            render_buffer,
            depth_buffer,
        );
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FragmentInput<const A: usize> {
    pub position: glam::Vec4,
    pub attributes: [f32; A],
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
