use glam::Vec4Swizzles;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Index, IndexMut};

pub mod presenter;

pub trait RenderBuffer: Index<[u32; 2], Output = Color> + IndexMut<[u32; 2]> {
    fn width(&self) -> u32;

    fn height(&self) -> u32;

    #[allow(unused_variables)]
    fn depth(&self, x: u32, y: u32) -> f32 {
        f32::NEG_INFINITY
    }

    #[allow(unused_variables)]
    fn set_depth(&mut self, x: u32, y: u32, depth: f32) {}
}

#[derive(Clone)]
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

    pub fn color_slice(&self) -> &[u8] {
        return bytemuck::cast_slice(self.color.as_slice());
    }

    fn calculate_index(&self, x: u32, y: u32) -> u32 {
        self.width * y + x
    }
}

impl Index<[u32; 2]> for ColorBuffer {
    type Output = Color;

    fn index(&self, index: [u32; 2]) -> &Self::Output {
        let index = self.calculate_index(index[0], index[1]) as usize;
        &self.color[index]
    }
}

impl IndexMut<[u32; 2]> for ColorBuffer {
    fn index_mut(&mut self, index: [u32; 2]) -> &mut Self::Output {
        let index = self.calculate_index(index[0], index[1]) as usize;
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
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
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

impl From<&glam::Vec4> for Color {
    fn from(vec: &glam::Vec4) -> Self {
        Self {
            r: vec.x,
            g: vec.y,
            b: vec.z,
            a: vec.w,
        }
    }
}

impl From<&Color> for glam::Vec4 {
    fn from(color: &Color) -> Self {
        color.to_vec4()
    }
}

impl From<&[f32; 4]> for Color {
    fn from(array: &[f32; 4]) -> Self {
        Self {
            r: array[0],
            g: array[1],
            b: array[2],
            a: array[3],
        }
    }
}

impl From<&Color> for [f32; 4] {
    fn from(color: &Color) -> Self {
        color.to_array()
    }
}

/// Pipeline defines how data is transformed to turn geometry into a rendered image ready for
/// presentation or further rendering
///
/// Data transformation is split into these stages:
/// - Vertex shading
/// - Shape assembley
/// - Clipping - I'm not yet sure about final position of clip stage in the pipeline
/// - Perspective division
/// - Fragment generation
/// - Fragment shading
///
/// It is very likely that number and order of stages will change as development progresses
pub struct Pipeline<VI, VS, SA, FG, R, FS, const A: usize, const S: usize>
where
    VI: Copy,
    VS: FnMut(VI) -> VertexOutput<A>,
    SA: ShapeAssembler<FragmentInput<A>, S>,
    R: FnMut([FragmentInput<A>; S]) -> Vec<FragmentInput<A>>,
    FG: FnMut(i32, i32) -> R,
    FS: FnMut(FragmentInput<A>) -> Color,
{
    vertex_shader: VS,
    shape_assembler: SA,
    fragment_generator: FG,
    fragment_shader: FS,

    vertex_input_pd: PhantomData<VI>,
}

impl<VI, VS, SA, FG, R, FS, const A: usize, const S: usize> Pipeline<VI, VS, SA, FG, R, FS, A, S>
where
    VI: Copy,
    VS: FnMut(VI) -> VertexOutput<A> + Copy,
    SA: ShapeAssembler<FragmentInput<A>, S>,
    R: FnMut([FragmentInput<A>; S]) -> Vec<FragmentInput<A>>,
    FG: FnMut(i32, i32) -> R,
    FS: FnMut(FragmentInput<A>) -> Color,
{
    pub fn new(vertex_shader: VS, shape_assembler: SA, fragment_generator: FG, fragment_shader: FS) -> Self {
        Self {
            vertex_shader,
            shape_assembler,
            fragment_generator,
            fragment_shader,

            vertex_input_pd: PhantomData,
        }
    }

    pub fn draw(
        &mut self,
        vertex_buffer: &[VI],
        framebuffer: &mut impl RenderBuffer,
    ) {
        let width = framebuffer.width();
        let height = framebuffer.height();
        let mut vertex_shader = self.vertex_shader;
        let viewport_transform = |vertex: VertexOutput<A>| {
            let inv_w = 1.0 / vertex.position.w;
            FragmentInput {
                position: (vertex.position.xyz() * inv_w, inv_w).into(),
                screen_position: glam::ivec2(
                    ((vertex.position.x + 1.0) * width as f32 / 2.0).round() as i32,
                    ((vertex.position.y + 1.0) * height as f32 / 2.0).round() as i32,
                ),
                attributes: vertex.attributes.map(|attribute| attribute * inv_w),
            }
        };
        vertex_buffer
            .into_iter()
            .map(|&vertex| vertex_shader(vertex))
            .map(viewport_transform)
            .assemble_shapes(self.shape_assembler)
            .map((self.fragment_generator)(width as i32, height as i32))
            .map(|fragments| fragments.into_iter())
            .flatten()
            .for_each(|fragment_input| {
                let position = fragment_input.screen_position.as_uvec2().to_array();
                framebuffer[position] = (self.fragment_shader)(fragment_input);
            });
    }
}

pub fn line_bresenham<const A: usize>(
    width: i32,
    height: i32,
) -> impl Fn([FragmentInput<A>; 2]) -> Vec<FragmentInput<A>> {
    move |shape: [FragmentInput<A>; 2]| {
        let [mut from, mut to] = shape;
        let [mut x0, mut y0] = from.screen_position.to_array();
        let [mut x1, mut y1] = to.screen_position.to_array();
        let run = x1 - x0;
        let rise = y1 - y0;
        let width_range = 0..width;
        let height_range = 0..height;
        let depth_range = 0.0..=1.0;
        let mut fragments = Vec::new();
        let mut check_and_add_fragment = |fragment: FragmentInput<A>| {
            let [x, y] = fragment.screen_position.to_array();
            let z = fragment.position.z;
            if width_range.contains(&x) & &height_range.contains(&y) & &depth_range.contains(&z) {
                fragments.push(fragment);
            }
        };
        if run == 0 {
            if y0 > y1 {
                mem::swap(&mut y0, &mut y1);
                mem::swap(&mut from, &mut to);
            }
            let mut interpolator = Interpolator::new(&from, &to, rise);
            for y in y0..=y1 {
                let (position, attributes) = interpolator.next().unwrap();
                check_and_add_fragment(FragmentInput {
                    position,
                    screen_position: glam::ivec2(x0, y),
                    attributes,
                });
            }
        } else {
            let slope = rise as f32 / run as f32;
            let adjust = if slope >= 0.0 { 1 } else { -1 };
            let mut offset = 0;
            if slope < 1.0 && slope > -1.0 {
                let delta = rise.abs() * 2;
                let mut threshold = run.abs();
                let threshold_inc = threshold * 2;
                let mut y;
                if x0 > x1 {
                    mem::swap(&mut x0, &mut x1);
                    mem::swap(&mut from, &mut to);
                    y = y1;
                } else {
                    y = y0;
                }
                let mut interpolator = Interpolator::new(&from, &to, run);
                for x in x0..=x1 {
                    let (position, attributes) = interpolator.next().unwrap();
                    check_and_add_fragment(FragmentInput {
                        position,
                        screen_position: glam::ivec2(x, y),
                        attributes,
                    });
                    offset += delta;
                    if offset >= threshold {
                        y += adjust;
                        threshold += threshold_inc;
                    }
                }
            } else {
                let delta = run.abs() * 2;
                let mut threshold = rise.abs();
                let threshold_inc = threshold * 2;
                let mut x;
                if y0 > y1 {
                    mem::swap(&mut y0, &mut y1);
                    mem::swap(&mut from, &mut to);
                    x = x1;
                } else {
                    x = x0;
                }
                let mut interpolator = Interpolator::new(&from, &to, rise);
                for y in y0..=y1 {
                    let (position, attributes) = interpolator.next().unwrap();
                    check_and_add_fragment(FragmentInput {
                        position,
                        screen_position: glam::ivec2(x, y),
                        attributes,
                    });
                    offset += delta;
                    if offset >= threshold {
                        x += adjust;
                        threshold += threshold_inc;
                    }
                }
            }
        }
        fragments
    }
}

#[derive(Copy, Clone)]
pub struct VertexOutput<const N: usize> {
    pub position: glam::Vec4,
    pub attributes: [f32; N],
}

#[derive(Copy, Clone)]
pub struct FragmentInput<const N: usize> {
    pub position: glam::Vec4,
    pub screen_position: glam::IVec2,
    pub attributes: [f32; N],
}

struct Interpolator<const N: usize> {
    from_pos: glam::Vec4,
    pos_delta: glam::Vec4,
    from_attrib: [f32; N],
    attrib_delta: [f32; N],
}

impl<const N: usize> Interpolator<N> {
    fn new(from: &FragmentInput<N>, to: &FragmentInput<N>, steps: i32) -> Self {
        let steps = steps as f32;

        let attrib_delta = {
            let mut attrib_delta = from.attributes;
            for ((&from, &to), delta) in from
                .attributes
                .iter()
                .zip(&to.attributes)
                .zip(&mut attrib_delta)
            {
                *delta = to - from / steps;
            }
            attrib_delta
        };

        Self {
            from_pos: from.position,
            pos_delta: (to.position - from.position) / steps,
            from_attrib: from.attributes,
            attrib_delta,
        }
    }
}

impl<const N: usize> Iterator for Interpolator<N> {
    type Item = (glam::Vec4, [f32; N]);

    fn next(&mut self) -> Option<Self::Item> {
        let w = 1.0 / self.from_pos.w;
        let position = (self.from_pos.xyz() * w, self.from_pos.w).into();
        let attributes = self.from_attrib.map(|attrib| attrib * w);
        let res = Some((position, attributes));
        self.from_pos += self.pos_delta;
        for (attr, delta) in self.from_attrib.iter_mut().zip(&self.attrib_delta) {
            *attr += delta;
        }
        res
    }
}

pub trait ShapeAssembler<V: Copy, const N: usize>: Copy {
    fn init(&mut self, iter: &mut impl Iterator<Item = V>);
    fn next(&mut self, iter: &mut impl Iterator<Item = V>) -> Option<[V; N]>;
    fn size_hint(&self, _: &impl Iterator<Item = V>) -> (usize, Option<usize>) {
        (0, None)
    }
}

#[derive(Copy, Clone)]
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

#[derive(Copy, Clone)]
pub struct ListShapeAssembpler<V: Copy, const N: usize> {
    shape: Option<[V; N]>,
}

impl<V: Copy, const N: usize> ListShapeAssembpler<V, N> {
    pub fn new() -> Self {
        Self { shape: None }
    }
}

impl<V: Copy, const N: usize> ShapeAssembler<V, N> for ListShapeAssembpler<V, N> {
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

struct ShapeAssemblerIterator<
    I: Iterator<Item = V>,
    SA: ShapeAssembler<V, N>,
    V: Copy,
    const N: usize,
> {
    iter: I,
    shape_assembler: SA,
}

impl<I: Iterator<Item = V>, SA: ShapeAssembler<V, N>, V: Copy, const N: usize>
    ShapeAssemblerIterator<I, SA, V, N>
{
    fn new(mut iter: I, mut shape_assembler: SA) -> Self {
        shape_assembler.init(&mut iter);
        Self {
            iter,
            shape_assembler,
        }
    }
}

impl<I: Iterator<Item = V>, SA: ShapeAssembler<V, N>, V: Copy, const N: usize> Iterator
    for ShapeAssemblerIterator<I, SA, V, N>
{
    type Item = [V; N];

    fn next(&mut self) -> Option<Self::Item> {
        self.shape_assembler.next(&mut self.iter)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.shape_assembler.size_hint(&self.iter)
    }
}

trait ShapeAssemblerIteratorTrait<SA: ShapeAssembler<V, N>, V: Copy, const N: usize>:
    Iterator<Item = V> + Sized
{
    fn assemble_shapes(self, shape_assembler: SA) -> ShapeAssemblerIterator<Self, SA, V, N> {
        ShapeAssemblerIterator::new(self, shape_assembler)
    }
}

impl<I: Iterator, SA: ShapeAssembler<I::Item, N>, const N: usize>
    ShapeAssemblerIteratorTrait<SA, I::Item, N> for I
where
    I::Item: Copy,
{
}
