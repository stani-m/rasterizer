use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use glam::Vec4Swizzles;

use rasterizer::Rasterizer;

pub mod clipper;
pub mod presenter;
pub mod rasterizer;

pub trait RenderBuffer: Index<[u32; 2], Output = Color> + IndexMut<[u32; 2]> {
    fn width(&self) -> u32;

    fn height(&self) -> u32;

    fn color_slice(&self) -> &[u8];

    #[allow(unused_variables)]
    fn depth(&self, x: u32, y: u32) -> f32 {
        f32::NEG_INFINITY
    }

    #[allow(unused_variables)]
    fn set_depth(&mut self, x: u32, y: u32, depth: f32) {}
}

#[derive(Clone, Debug)]
pub struct ColorBuffer {
    color: Vec<Color>,
    width: u32,
    height: u32,
}

impl ColorBuffer {
    pub fn new(width: u32, height: u32) -> Self {
        let size = width * height;
        Self {
            color: vec![Color::default(); size as usize],
            width,
            height,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.color
            .resize((width * height) as usize, Color::default());
    }

    pub fn clear(&mut self, color: Color) {
        self.color.fill(color);
    }

    fn calculate_index(&self, x: u32, y: u32) -> usize {
        (self.width * y + x) as usize
    }
}

impl Index<[u32; 2]> for ColorBuffer {
    type Output = Color;

    #[inline]
    fn index(&self, index: [u32; 2]) -> &Self::Output {
        let index = self.calculate_index(index[0], index[1]);
        &self.color[index]
    }
}

impl IndexMut<[u32; 2]> for ColorBuffer {
    #[inline]
    fn index_mut(&mut self, index: [u32; 2]) -> &mut Self::Output {
        let index = self.calculate_index(index[0], index[1]);
        &mut self.color[index]
    }
}

impl RenderBuffer for ColorBuffer {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn color_slice(&self) -> &[u8] {
        return bytemuck::cast_slice(self.color.as_slice());
    }
}

#[derive(Clone, Debug)]
pub struct ColorDepthBuffer {
    color: Vec<Color>,
    depth: Vec<f32>,
    width: u32,
    height: u32,
}

impl ColorDepthBuffer {
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            color: vec![Color::default(); size],
            depth: vec![f32::INFINITY; size],
            width,
            height,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let size = (width * height) as usize;
        self.color.resize(size, Color::default());
        self.depth.resize(size, f32::INFINITY);
    }

    pub fn clear_color(&mut self, color: Color) {
        self.color.fill(color);
    }

    pub fn clear_depth(&mut self, depth: f32) {
        self.depth.fill(depth);
    }

    fn calculate_index(&self, x: u32, y: u32) -> usize {
        (self.width * y + x) as usize
    }
}

impl Index<[u32; 2]> for ColorDepthBuffer {
    type Output = Color;

    #[inline]
    fn index(&self, index: [u32; 2]) -> &Self::Output {
        let index = self.calculate_index(index[0], index[1]);
        &self.color[index]
    }
}

impl IndexMut<[u32; 2]> for ColorDepthBuffer {
    #[inline]
    fn index_mut(&mut self, index: [u32; 2]) -> &mut Self::Output {
        let index = self.calculate_index(index[0], index[1]);
        &mut self.color[index]
    }
}

impl RenderBuffer for ColorDepthBuffer {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn color_slice(&self) -> &[u8] {
        return bytemuck::cast_slice(self.color.as_slice());
    }

    #[inline]
    fn depth(&self, x: u32, y: u32) -> f32 {
        let index = self.calculate_index(x, y);
        self.depth[index]
    }

    #[inline]
    fn set_depth(&mut self, x: u32, y: u32, depth: f32) {
        let index = self.calculate_index(x, y);
        self.depth[index] = depth;
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

/// Pipeline defines how data is transformed to turn geometry into a rendered image ready for
/// presentation or further rendering
///
/// Data transformation is split into these stages:
/// - Vertex shading
/// - Shape assembly
/// - Clipping
/// - Perspective division
/// - Rasterization
/// - Fragment shading
///
/// It is possible that number and order of stages will change as development progresses
pub struct Pipeline<VI, U, VS, SA, C, R, DT, FS, const A: usize, const S: usize>
where
    VI: Copy,
    VS: FnMut(VI, &U) -> VertexOutput<A> + Copy,
    SA: ShapeAssembler<u32, S>,
    C: FnMut([VertexOutput<A>; S]) -> Vec<[VertexOutput<A>; S]>,
    R: Rasterizer<A, S>,
    DT: FnMut(f32, f32) -> bool,
    FS: FnMut(FragmentInput<A>, &U) -> Color,
{
    vertex_shader: VS,
    shape_assembler: SA,
    clipper: C,
    rasterizer: R,
    depth_test: DT,
    fragment_shader: FS,

    vertex_input_pd: PhantomData<VI>,
    uniform_pd: PhantomData<U>,
}

impl<VI, U, VS, SA, C, R, DT, FS, const A: usize, const S: usize>
    Pipeline<VI, U, VS, SA, C, R, DT, FS, A, S>
where
    VI: Copy,
    VS: FnMut(VI, &U) -> VertexOutput<A> + Copy,
    SA: ShapeAssembler<u32, S>,
    C: FnMut([VertexOutput<A>; S]) -> Vec<[VertexOutput<A>; S]>,
    R: Rasterizer<A, S>,
    DT: FnMut(f32, f32) -> bool,
    FS: FnMut(FragmentInput<A>, &U) -> Color,
{
    pub fn new(
        vertex_shader: VS,
        shape_assembler: SA,
        clipper: C,
        rasterizer: R,
        depth_test: DT,
        fragment_shader: FS,
    ) -> Self {
        Self {
            vertex_shader,
            shape_assembler,
            clipper,
            rasterizer,
            depth_test,
            fragment_shader,

            vertex_input_pd: PhantomData,
            uniform_pd: PhantomData,
        }
    }

    pub fn draw_indexed(
        &mut self,
        vertex_buffer: &[VI],
        index_buffer: &[u32],
        uniforms: &U,
        framebuffer: &mut impl RenderBuffer,
    ) {
        let width = framebuffer.width();
        let height = framebuffer.height();

        let perspective_divide = |shape: [VertexOutput<A>; S]| {
            shape.map(|vertex| {
                let inv_w = 1.0 / vertex.position.w;
                let position = (vertex.position.xyz() * inv_w, inv_w).into();
                FragmentInput {
                    position,
                    screen_position: glam::ivec2(
                        ((position.x + 1.0) * width as f32 / 2.0).round() as i32,
                        ((position.y + 1.0) * height as f32 / 2.0).round() as i32,
                    ),
                    attributes: vertex.attributes.map(|attribute| attribute * inv_w),
                }
            })
        };

        let mut rasterizer_action = |fragment_input: FragmentInput<A>| {
            let position = fragment_input.screen_position.as_uvec2().to_array();
            let depth_test_passed = (self.depth_test)(
                framebuffer.depth(position[0], position[1]),
                fragment_input.position.z,
            );
            if depth_test_passed {
                framebuffer[position] = (self.fragment_shader)(fragment_input, &uniforms);
                framebuffer.set_depth(position[0], position[1], fragment_input.position.z);
            }
        };

        self.rasterizer.set_screen_size(width, height);
        let shaded_vertices = vertex_buffer
            .into_iter()
            .map(|&vertex| (self.vertex_shader)(vertex, &uniforms))
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

    pub fn draw(&mut self, vertex_buffer: &[VI], unforms: &U, framebuffer: &mut impl RenderBuffer) {
        let index_buffer = (0..vertex_buffer.len() as u32).collect::<Vec<_>>();
        self.draw_indexed(vertex_buffer, &index_buffer, unforms, framebuffer);
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VertexOutput<const N: usize> {
    pub position: glam::Vec4,
    pub attributes: [f32; N],
}

#[derive(Copy, Clone, Debug)]
pub struct FragmentInput<const N: usize> {
    pub position: glam::Vec4,
    pub screen_position: glam::IVec2,
    pub attributes: [f32; N],
}

pub trait ShapeAssembler<V: Copy, const S: usize>: Copy {
    fn init(&mut self, iter: &mut impl Iterator<Item = V>);
    fn next(&mut self, iter: &mut impl Iterator<Item = V>) -> Option<[V; S]>;
    fn size_hint(&self, _: &impl Iterator<Item = V>) -> (usize, Option<usize>) {
        (0, None)
    }
}

#[derive(Copy, Clone)]
pub struct StripShapeAssembler<V: Copy, const S: usize> {
    shape: Option<[V; S]>,
}

impl<V: Copy, const S: usize> StripShapeAssembler<V, S> {
    pub fn new() -> Self {
        Self { shape: None }
    }
}

impl<V: Copy, const S: usize> ShapeAssembler<V, S> for StripShapeAssembler<V, S> {
    fn init(&mut self, iter: &mut impl Iterator<Item = V>) {
        let mut shape = Vec::with_capacity(S);
        for _ in 0..S {
            if let Some(vertex) = iter.next() {
                shape.push(vertex);
            }
        }
        self.shape = shape.try_into().ok();
    }

    fn next(&mut self, iter: &mut impl Iterator<Item = V>) -> Option<[V; S]> {
        if let Some(ref mut shape) = self.shape {
            let res = *shape;
            if let Some(vertex) = iter.next() {
                for i in 1..S {
                    shape[i - 1] = shape[i];
                }
                shape[S - 1] = vertex;
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

#[derive(Copy, Clone)]
pub struct ListShapeAssembler<V: Copy, const S: usize> {
    shape: Option<[V; S]>,
}

impl<V: Copy, const S: usize> ListShapeAssembler<V, S> {
    pub fn new() -> Self {
        Self { shape: None }
    }
}

impl<V: Copy, const S: usize> ShapeAssembler<V, S> for ListShapeAssembler<V, S> {
    fn init(&mut self, iter: &mut impl Iterator<Item = V>) {
        let mut shape = Vec::with_capacity(S);
        for _ in 0..S {
            if let Some(vertex) = iter.next() {
                shape.push(vertex);
            }
        }
        self.shape = shape.try_into().ok();
    }

    fn next(&mut self, iter: &mut impl Iterator<Item = V>) -> Option<[V; S]> {
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

struct ShapeAssemblerIterator<I, SA, V, const S: usize>
where
    I: Iterator<Item = V>,
    SA: ShapeAssembler<V, S>,
    V: Copy,
{
    iter: I,
    shape_assembler: SA,
}

impl<I, SA, V, const S: usize> ShapeAssemblerIterator<I, SA, V, S>
where
    I: Iterator<Item = V>,
    SA: ShapeAssembler<V, S>,
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

impl<I, SA, V, const S: usize> Iterator for ShapeAssemblerIterator<I, SA, V, S>
where
    I: Iterator<Item = V>,
    SA: ShapeAssembler<V, S>,
    V: Copy,
{
    type Item = [V; S];

    fn next(&mut self) -> Option<Self::Item> {
        self.shape_assembler.next(&mut self.iter)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.shape_assembler.size_hint(&self.iter)
    }
}

trait ShapeAssemblerIteratorTrait<SA, V, const S: usize>: Iterator<Item = V> + Sized
where
    SA: ShapeAssembler<V, S>,
    V: Copy,
{
    fn assemble_shapes(self, shape_assembler: SA) -> ShapeAssemblerIterator<Self, SA, V, S> {
        ShapeAssemblerIterator::new(self, shape_assembler)
    }
}

impl<I, SA, const S: usize> ShapeAssemblerIteratorTrait<SA, I::Item, S> for I
where
    I: Iterator,
    I::Item: Copy,
    SA: ShapeAssembler<I::Item, S>,
{
}
