use std::mem;
use std::ops::{Add, Mul, Range};

use glam::Vec4Swizzles;
use rayon_core::{Scope, ThreadPool, ThreadPoolBuilder};

use crate::{Buffer, DepthState, Multisampler, UnshadedFragment};

pub trait Rasterizer<const N: usize, const A: usize, const S: usize> {
    fn set_framebuffer_size(&mut self, width: u32, height: u32);
    fn rasterize<U, T, FS, B, MS>(
        &self,
        shape: [UnshadedFragment<A>; N],
        fragment_tools: &mut FragmentTools<U, T, FS, B, A, S>,
        depth_tools: &mut Option<DepthTools<S>>,
        multisampler: &MS,
    ) where
        U: Copy + Send,
        T: Copy + Default + Send,
        FS: Fn(UnshadedFragment<A>, U) -> T + Sync,
        B: Fn(&T, &T) -> T + Sync,
        MS: Multisampler<S> + Sync;
}

pub struct FragmentTools<'a, U, T, FS, B, const A: usize, const S: usize> {
    pub(crate) uniforms: U,
    pub(crate) fragment_shader: &'a FS,
    pub(crate) blend_function: &'a B,
    pub(crate) render_buffer: &'a mut Buffer<T, S>,
}

impl<'a, U, T, FS, B, const A: usize, const S: usize> FragmentTools<'a, U, T, FS, B, A, S>
where
    U: Copy,
    T: Copy + Default,
    FS: Fn(UnshadedFragment<A>, U) -> T,
    B: Fn(&T, &T) -> T,
{
    pub fn shade_fragment(&self, fragment: UnshadedFragment<A>) -> T {
        (self.fragment_shader)(fragment, self.uniforms)
    }

    pub fn blend_and_write(&mut self, x: u32, y: u32, sample: u32, value: &T) {
        let dst_value = &self.render_buffer[[x, y, sample]];
        self.render_buffer[[x, y, sample]] = (self.blend_function)(value, dst_value);
    }
}

struct UnsafeFragmentTools<'a, U, T, FS, B, const A: usize, const S: usize>(
    *mut FragmentTools<'a, U, T, FS, B, A, S>,
)
where
    U: Send,
    T: Send,
    FS: Sync,
    B: Sync;

unsafe impl<'a, U, T, FS, B, const A: usize, const S: usize> Send
    for UnsafeFragmentTools<'a, U, T, FS, B, A, S>
where
    U: Send,
    T: Send,
    FS: Sync,
    B: Sync,
{
}

unsafe impl<'a, U, T, FS, B, const A: usize, const S: usize> Sync
    for UnsafeFragmentTools<'a, U, T, FS, B, A, S>
where
    U: Send,
    T: Send,
    FS: Sync,
    B: Sync,
{
}

pub struct DepthTools<'a, const S: usize> {
    pub(crate) depth_state: &'a DepthState,
    pub(crate) depth_buffer: &'a mut Buffer<f32, S>,
}

impl<'a, const S: usize> DepthTools<'a, S> {
    pub fn compare(&self, x: u32, y: u32, sample: u32, depth: f32) -> bool {
        let dst_depth = self.depth_buffer[[x, y, sample]];
        (self.depth_state.depth_function)(&depth, &dst_depth)
    }

    pub fn write_depth(&mut self, x: u32, y: u32, sample: u32, depth: f32) {
        self.depth_buffer[[x, y, sample]] = depth;
    }
}

struct UnsafeDepthTools<'a, const S: usize>(*mut Option<DepthTools<'a, S>>);
unsafe impl<'a, const S: usize> Send for UnsafeDepthTools<'a, S> {}
unsafe impl<'a, const S: usize> Sync for UnsafeDepthTools<'a, S> {}

#[derive(Copy, Clone, Debug)]
pub struct BresenhamLineRasterizer {
    width: i32,
    height: i32,
}

impl BresenhamLineRasterizer {
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
        }
    }
}

impl<const A: usize, const S: usize> Rasterizer<2, A, S> for BresenhamLineRasterizer {
    fn set_framebuffer_size(&mut self, width: u32, height: u32) {
        self.width = width as i32;
        self.height = height as i32;
    }

    fn rasterize<U, T, FS, B, MS>(
        &self,
        line: [UnshadedFragment<A>; 2],
        fragment_tools: &mut FragmentTools<U, T, FS, B, A, S>,
        depth_tools: &mut Option<DepthTools<S>>,
        _: &MS,
    ) where
        U: Copy,
        T: Copy + Default,
        FS: Fn(UnshadedFragment<A>, U) -> T,
        B: Fn(&T, &T) -> T,
    {
        let [mut from, mut to] = line;
        let [mut x0, mut y0] = from.position.xy().round().as_ivec2().to_array();
        let [mut x1, mut y1] = to.position.xy().round().as_ivec2().to_array();
        let mut run = x1 - x0;
        let mut rise = y1 - y0;
        let width_range = 0..self.width;
        let height_range = 0..self.height;
        let depth_range = 0.0..=1.0;
        let mut check_and_perform_action =
            |x: i32, y: i32, zw: glam::Vec2, attributes: [f32; A]| {
                let z = zw[0];

                if width_range.contains(&x) && height_range.contains(&y) && depth_range.contains(&z)
                {
                    let fragment_value = fragment_tools.shade_fragment(UnshadedFragment {
                        position: glam::vec4(x as f32 + 0.5, y as f32 + 0.5, z, zw[1]),
                        attributes,
                    });
                    for sample in 0..(S as u32) {
                        if let Some(depth_tools) = depth_tools {
                            let depth_test_passed =
                                depth_tools.compare(x as u32, y as u32, sample, z);
                            if depth_test_passed {
                                fragment_tools.blend_and_write(
                                    x as u32,
                                    y as u32,
                                    sample,
                                    &fragment_value,
                                );

                                if depth_tools.depth_state.write_depth {
                                    depth_tools.write_depth(x as u32, y as u32, sample, z);
                                }
                            }
                        } else {
                            fragment_tools.blend_and_write(
                                x as u32,
                                y as u32,
                                sample,
                                &fragment_value,
                            );
                        }
                    }
                }
            };
        if run == 0 {
            if y0 > y1 {
                mem::swap(&mut y0, &mut y1);
                mem::swap(&mut from, &mut to);
                rise = -rise;
            }
            let mut interpolator = Interpolator::new(&from, &to, rise);
            for y in y0..=y1 {
                let (zw, attributes) = interpolator.next().unwrap();
                check_and_perform_action(x0, y, zw, attributes);
            }
        } else {
            let slope = rise as f32 / run as f32;
            let direction = if slope >= 0.0 { 1 } else { -1 };
            let mut offset = 0;
            if slope.abs() < 1.0 {
                let delta = rise.abs() * 2;
                let mut threshold = run.abs();
                let increment = threshold * 2;
                let mut y;
                if x0 > x1 {
                    mem::swap(&mut x0, &mut x1);
                    mem::swap(&mut from, &mut to);
                    run = -run;
                    y = y1;
                } else {
                    y = y0;
                }
                let mut interpolator = Interpolator::new(&from, &to, run);
                for x in x0..=x1 {
                    let (zw, attributes) = interpolator.next().unwrap();
                    check_and_perform_action(x, y, zw, attributes);
                    offset += delta;
                    if offset >= threshold {
                        y += direction;
                        threshold += increment;
                    }
                }
            } else {
                let delta = run.abs() * 2;
                let mut threshold = rise.abs();
                let increment = threshold * 2;
                let mut x;
                if y0 > y1 {
                    mem::swap(&mut y0, &mut y1);
                    mem::swap(&mut from, &mut to);
                    rise = -rise;
                    x = x1;
                } else {
                    x = x0;
                }
                let mut interpolator = Interpolator::new(&from, &to, rise);
                for y in y0..=y1 {
                    let (zw, attributes) = interpolator.next().unwrap();
                    check_and_perform_action(x, y, zw, attributes);
                    offset += delta;
                    if offset >= threshold {
                        x += direction;
                        threshold += increment;
                    }
                }
            }
        }
    }
}

struct Interpolator<const A: usize> {
    from_pos: glam::Vec2,
    pos_delta: glam::Vec2,
    from_attrib: [f32; A],
    attrib_delta: [f32; A],
}

impl<const A: usize> Interpolator<A> {
    fn new(from: &UnshadedFragment<A>, to: &UnshadedFragment<A>, steps: i32) -> Self {
        Self::from_data(
            from.position.zw(),
            to.position.zw(),
            from.attributes,
            to.attributes,
            steps,
        )
    }

    fn from_data(
        from_pos: glam::Vec2,
        to_pos: glam::Vec2,
        from_attrib: [f32; A],
        to_attrib: [f32; A],
        steps: i32,
    ) -> Self {
        let steps = steps as f32;

        let attrib_delta = {
            let mut attrib_delta = [0.0; A];
            for ((&from, &to), delta) in from_attrib.iter().zip(&to_attrib).zip(&mut attrib_delta) {
                *delta = (to - from) / steps;
            }
            attrib_delta
        };

        Self {
            from_pos,
            pos_delta: (to_pos - from_pos) / steps,
            from_attrib,
            attrib_delta,
        }
    }

    fn step(&mut self) {
        self.from_pos += self.pos_delta;
        for (attr, delta) in self.from_attrib.iter_mut().zip(&self.attrib_delta) {
            *attr += delta;
        }
    }
}

impl<const A: usize> Iterator for Interpolator<A> {
    type Item = (glam::Vec2, [f32; A]);

    fn next(&mut self) -> Option<Self::Item> {
        let w = 1.0 / self.from_pos[1];
        let attributes = self.from_attrib.map(|attrib| (attrib * w) as f32);
        let res = Some((self.from_pos, attributes));
        self.step();
        res
    }
}

#[derive(Debug)]
pub struct AALineRasterizer {
    framebuffer_bounds: Bounds,
}

impl AALineRasterizer {
    pub fn new() -> Self {
        Self {
            framebuffer_bounds: Bounds {
                min: glam::uvec2(0, 0),
                max: glam::uvec2(0, 0),
            },
        }
    }
}

impl<const A: usize, const S: usize> Rasterizer<2, A, S> for AALineRasterizer {
    fn set_framebuffer_size(&mut self, width: u32, height: u32) {
        self.framebuffer_bounds.max = glam::uvec2(width, height);
    }

    fn rasterize<U, T, FS, B, MS>(
        &self,
        line: [UnshadedFragment<A>; 2],
        fragment_tools: &mut FragmentTools<U, T, FS, B, A, S>,
        depth_tools: &mut Option<DepthTools<S>>,
        multisampler: &MS,
    ) where
        U: Copy + Send,
        T: Copy + Default + Send,
        FS: Fn(UnshadedFragment<A>, U) -> T + Sync,
        B: Fn(&T, &T) -> T + Sync,
        MS: Multisampler<S> + Sync,
    {
        let [v0, v1] = line;
        let vector = (v1.position.xy() - v0.position.xy()).normalize();
        let half_normal = glam::vec4(-vector.y, vector.x, 0.0, 0.0) * 0.5;

        let v00 = UnshadedFragment {
            position: v0.position + half_normal,
            ..v0
        };
        let v01 = UnshadedFragment {
            position: v0.position - half_normal,
            ..v0
        };
        let v10 = UnshadedFragment {
            position: v1.position + half_normal,
            ..v1
        };
        let v11 = UnshadedFragment {
            position: v1.position - half_normal,
            ..v1
        };

        let triangles = [[v00, v01, v10], [v01, v10, v11]];

        let double_area = calculate_double_area(&triangles[0]);

        let bounds = Bounds::from(&[v00, v01, v10, v11]).intersection(self.framebuffer_bounds);

        let inv_area = -1.0 / double_area;

        let static_data = [
            StaticData::new(&triangles[0], inv_area),
            StaticData::new(&triangles[1], -inv_area),
        ];

        let fragment_tools = &UnsafeFragmentTools(fragment_tools);
        let depth_tools = &UnsafeDepthTools(depth_tools);

        let mut from = v0.position.xy().as_ivec2();
        let mut to = v1.position.xy().as_ivec2();

        let run = to.x - from.x;
        let rise = to.y - from.y;

        if run == 0 || rise == 0 {
            process_bounds(
                &bounds,
                &triangles,
                fragment_tools,
                depth_tools,
                multisampler,
                &static_data,
                false,
            );
        } else {
            let slope = rise as f32 / run as f32;
            let direction = if slope >= 0.0 { 1 } else { -1 };
            let mut offset = 0;
            if slope.abs() < 1.0 {
                let delta = rise.abs() * 2;
                let mut threshold = run.abs();
                let increment = threshold * 2;
                if from.x > to.x {
                    mem::swap(&mut from, &mut to);
                }
                let mut y = from.y;
                for x in from.x..=to.x {
                    let segment =
                        bounds.intersection_i32(glam::ivec2(x, y - 1), glam::ivec2(x + 1, y + 2));

                    process_bounds(
                        &segment,
                        &triangles,
                        fragment_tools,
                        depth_tools,
                        multisampler,
                        &static_data,
                        false,
                    );

                    offset += delta;
                    if offset >= threshold {
                        y += direction;
                        threshold += increment;
                    }
                }
            } else {
                let delta = run.abs() * 2;
                let mut threshold = rise.abs();
                let increment = threshold * 2;
                if from.y > to.y {
                    mem::swap(&mut from, &mut to);
                }
                let mut x = from.x;
                for y in from.y..=to.y {
                    if (bounds.min.y..bounds.max.y).contains(&(y as u32)) {
                        let x_min = (x - 1).clamp(bounds.min.x as i32, bounds.max.x as i32) as u32;
                        let x_max = (x + 2).clamp(bounds.min.x as i32, bounds.max.x as i32) as u32;

                        let barycentric_computations = multisampler.y_offsets().map(|y_offset| {
                            static_data.map(|static_data| {
                                static_data.barycentric_computations(y as f32 + y_offset)
                            })
                        });

                        process_row(
                            x_min..x_max,
                            &triangles,
                            fragment_tools,
                            depth_tools,
                            multisampler,
                            &static_data,
                            &barycentric_computations,
                            false,
                            y as u32,
                        );
                    }
                    offset += delta;
                    if offset >= threshold {
                        x += direction;
                        threshold += increment;
                    }
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Face {
    Cw,
    Ccw,
}

#[derive(Copy, Clone, Debug)]
pub struct EdgeFunctionRasterizer {
    framebuffer_bounds: Bounds,
    cull_face: Option<Face>,
}

impl EdgeFunctionRasterizer {
    pub fn new(cull_face: Option<Face>) -> Self {
        Self {
            framebuffer_bounds: Bounds {
                min: glam::uvec2(0, 0),
                max: glam::uvec2(0, 0),
            },
            cull_face,
        }
    }
}

impl<const A: usize, const S: usize> Rasterizer<3, A, S> for EdgeFunctionRasterizer {
    fn set_framebuffer_size(&mut self, width: u32, height: u32) {
        self.framebuffer_bounds.max = glam::uvec2(width, height);
    }

    fn rasterize<U, T, FS, B, MS>(
        &self,
        triangle: [UnshadedFragment<A>; 3],
        fragment_tools: &mut FragmentTools<U, T, FS, B, A, S>,
        depth_tools: &mut Option<DepthTools<S>>,
        multisampler: &MS,
    ) where
        U: Copy + Send,
        T: Copy + Default + Send,
        FS: Fn(UnshadedFragment<A>, U) -> T + Sync,
        B: Fn(&T, &T) -> T + Sync,
        MS: Multisampler<S> + Sync,
    {
        let double_area = calculate_double_area(&triangle);

        if let Some(cull_face) = self.cull_face {
            if skip_triangle(double_area, cull_face) {
                return;
            }
        }

        let bounds = Bounds::from(&triangle).intersection(self.framebuffer_bounds);

        let inv_area = -1.0 / double_area;

        let static_data = [StaticData::new(&triangle, inv_area)];

        // DepthTools and FragmentTools don't need to be unsafe here but it makes code reuse much easier
        let fragment_tools = &UnsafeFragmentTools(fragment_tools);
        let depth_tools = &UnsafeDepthTools(depth_tools);

        let triangles = [triangle];

        process_bounds(
            &bounds,
            &triangles,
            fragment_tools,
            depth_tools,
            multisampler,
            &static_data,
            false,
        );
    }
}

#[derive(Debug)]
pub struct EdgeFunctionMTRasterizer {
    framebuffer_bounds: Bounds,
    cull_face: Option<Face>,
    thread_pool: ThreadPool,
}

impl EdgeFunctionMTRasterizer {
    pub fn new(cull_face: Option<Face>, thread_count: usize) -> Self {
        Self {
            framebuffer_bounds: Bounds::default(),
            cull_face,
            thread_pool: ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .unwrap(),
        }
    }
}

impl<const A: usize, const S: usize> Rasterizer<3, A, S> for EdgeFunctionMTRasterizer {
    fn set_framebuffer_size(&mut self, width: u32, height: u32) {
        self.framebuffer_bounds.max = glam::uvec2(width, height);
    }

    fn rasterize<U, T, FS, B, MS>(
        &self,
        triangle: [UnshadedFragment<A>; 3],
        fragment_tools: &mut FragmentTools<U, T, FS, B, A, S>,
        depth_tools: &mut Option<DepthTools<S>>,
        multisampler: &MS,
    ) where
        U: Copy + Send,
        T: Copy + Default + Send,
        FS: Fn(UnshadedFragment<A>, U) -> T + Sync,
        B: Fn(&T, &T) -> T + Sync,
        MS: Multisampler<S> + Sync,
    {
        let double_area = calculate_double_area(&triangle);

        if let Some(cull_face) = self.cull_face {
            if skip_triangle(double_area, cull_face) {
                return;
            }
        }

        let bounds = Bounds::from(&triangle).intersection(self.framebuffer_bounds);

        let inv_area = -1.0 / double_area;

        let static_data = [StaticData::new(&triangle, inv_area)];

        let fragment_tools = &UnsafeFragmentTools(fragment_tools);
        let depth_tools = &UnsafeDepthTools(depth_tools);

        let triangles = [triangle];

        self.thread_pool.in_place_scope(|scope| {
            for y in bounds.min.y..bounds.max.y {
                scope.spawn(move |_| {
                    let barycentric_computations = multisampler.y_offsets().map(|y_offset| {
                        static_data.map(|static_data| {
                            static_data.barycentric_computations(y as f32 + y_offset)
                        })
                    });

                    process_row(
                        bounds.min.x..bounds.max.x,
                        &triangles,
                        fragment_tools,
                        depth_tools,
                        multisampler,
                        &static_data,
                        &barycentric_computations,
                        false,
                        y,
                    );
                });
            }
        });
    }
}

#[derive(Debug)]
pub struct EdgeFunctionTiledRasterizer<const TS: usize> {
    framebuffer_bounds: Bounds,
    cull_face: Option<Face>,
    thread_pool: ThreadPool,
}

impl<const TS: usize> EdgeFunctionTiledRasterizer<TS> {
    pub fn new(cull_face: Option<Face>, thread_count: usize) -> Self {
        assert_ne!(TS, 0, "tile size cannot be zero");
        Self {
            framebuffer_bounds: Bounds::default(),
            cull_face,
            thread_pool: ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .unwrap(),
        }
    }
}

impl<const A: usize, const S: usize, const TS: usize> Rasterizer<3, A, S>
    for EdgeFunctionTiledRasterizer<TS>
{
    fn set_framebuffer_size(&mut self, width: u32, height: u32) {
        self.framebuffer_bounds.max = glam::uvec2(width, height);
    }

    fn rasterize<U, T, FS, B, MS>(
        &self,
        triangle: [UnshadedFragment<A>; 3],
        fragment_tools: &mut FragmentTools<U, T, FS, B, A, S>,
        depth_tools: &mut Option<DepthTools<S>>,
        multisampler: &MS,
    ) where
        U: Copy + Send,
        T: Copy + Default + Send,
        FS: Fn(UnshadedFragment<A>, U) -> T + Sync,
        B: Fn(&T, &T) -> T + Sync,
        MS: Multisampler<S> + Sync,
    {
        let double_area = calculate_double_area(&triangle);

        if let Some(cull_face) = self.cull_face {
            if skip_triangle(double_area, cull_face) {
                return;
            }
        }

        let original_bounds = Bounds::from(&triangle);
        let bounds = original_bounds.intersection(self.framebuffer_bounds);
        let y_max_aligned = bounds.max.y - (bounds.max.y - bounds.min.y) % TS as u32;

        let inv_area = -1.0 / double_area;

        let static_data = [StaticData::new(&triangle, inv_area)];

        let lines = [
            Line::new(triangle[0].position.xy(), triangle[1].position.xy()),
            Line::new(triangle[1].position.xy(), triangle[2].position.xy()),
            Line::new(triangle[2].position.xy(), triangle[0].position.xy()),
        ];

        let fragment_tools = &UnsafeFragmentTools(fragment_tools);
        let depth_tools = &UnsafeDepthTools(depth_tools);

        let triangles = [triangle];

        self.thread_pool.in_place_scope(|scope| {
            for y_min in (bounds.min.y..y_max_aligned).step_by(TS) {
                let mut tile_row = bounds;
                tile_row.min.y = y_min;
                tile_row.max.y = y_min + TS as u32;

                process_bounds_tiled::<U, T, FS, B, MS, A, S, 1, TS>(
                    scope,
                    tile_row,
                    &triangles,
                    fragment_tools,
                    depth_tools,
                    multisampler,
                    &static_data,
                    &lines,
                );
            }
            let mut tile_row = bounds;
            tile_row.min.y = y_max_aligned;

            process_bounds_tiled::<U, T, FS, B, MS, A, S, 1, TS>(
                scope,
                tile_row,
                &triangles,
                fragment_tools,
                depth_tools,
                multisampler,
                &static_data,
                &lines,
            );
        });
    }
}

#[derive(Copy, Clone, Default, Debug)]
struct Bounds {
    min: glam::UVec2,
    max: glam::UVec2,
}

impl Bounds {
    fn intersect_or_contain<const A: usize>(
        &self,
        lines: &[Line; 3],
        triangle: &[UnshadedFragment<A>; 3],
    ) -> bool {
        let min_bound = self.min.as_vec2();
        let max_bound = self.max.as_vec2();
        let x_range = min_bound.x..=max_bound.x;
        let y_range = min_bound.y..=max_bound.y;
        let mut contain = true;
        for (line, point) in lines.into_iter().zip(triangle.into_iter()) {
            if !x_range.contains(&point.position.x) || !y_range.contains(&point.position.y) {
                contain = false;
            }
            let x1 = line.solve_x(min_bound.y);
            let x2 = line.solve_x(max_bound.y);
            let y1 = line.solve_y(min_bound.x);
            let y2 = line.solve_y(max_bound.x);
            if x_range.contains(&x1)
                || x_range.contains(&x2)
                || y_range.contains(&y1)
                || y_range.contains(&y2)
            {
                return true;
            }
        }
        contain
    }

    fn middle(&self) -> glam::Vec2 {
        (self.min.as_vec2() + self.max.as_vec2()) * 0.5
    }

    fn intersection(self, other: Self) -> Self {
        Bounds {
            min: self.min.max(other.min).min(other.max),
            max: self.max.min(other.max).max(other.min),
        }
    }

    fn intersection_i32(self, min: glam::IVec2, max: glam::IVec2) -> Self {
        let min = self.min.as_ivec2().max(min);
        Bounds {
            min: min.as_uvec2(),
            max: self.max.as_ivec2().min(max).max(min).as_uvec2(),
        }
    }
}

impl<const V: usize, const A: usize> From<&[UnshadedFragment<A>; V]> for Bounds {
    fn from(shape: &[UnshadedFragment<A>; V]) -> Self {
        let mut min_bound = shape[0].position.xy();
        let mut max_bound = shape[0].position.xy();
        for vertex in shape.iter().map(|vertex| &vertex.position) {
            if vertex.x < min_bound.x {
                min_bound.x = vertex.x;
            }
            if vertex.y < min_bound.y {
                min_bound.y = vertex.y;
            }
            if vertex.x > max_bound.x {
                max_bound.x = vertex.x;
            }
            if vertex.y > max_bound.y {
                max_bound.y = vertex.y;
            }
        }
        Bounds {
            min: min_bound.max(glam::vec2(0.0, 0.0)).as_uvec2(),
            max: max_bound.ceil().as_uvec2(),
        }
    }
}

#[derive(Copy, Clone)]
struct Line {
    slope: f32,
    inv_slope: f32,
    intercept: f32,
    inv_intercept: f32,
}

impl Line {
    fn new(a: glam::Vec2, b: glam::Vec2) -> Self {
        let run = b.x - a.x;
        let rise = b.y - a.y;
        let slope = rise / run;
        let inv_slope = run / rise;
        Self {
            slope,
            inv_slope,
            intercept: a.y - slope * a.x,
            inv_intercept: a.x - inv_slope * a.y,
        }
    }

    fn solve_y(&self, x: f32) -> f32 {
        self.slope * x + self.intercept
    }

    fn solve_x(&self, y: f32) -> f32 {
        self.inv_slope * y + self.inv_intercept
    }
}

#[derive(Copy, Clone)]
struct StaticDataComponent {
    base: f64,
    x_component: f64,
    y_component: f64,
}

impl StaticDataComponent {
    fn barycentric_computation(&self, y: f32) -> BarycentricComputation {
        BarycentricComputation {
            value: self.base + self.y_component * y as f64,
            x_delta: self.x_component,
        }
    }
}

#[derive(Copy, Clone)]
struct StaticData(
    StaticDataComponent,
    StaticDataComponent,
    StaticDataComponent,
);

impl StaticData {
    fn new<const A: usize>(triangle: &[UnshadedFragment<A>; 3], inv_area: f32) -> Self {
        let v0 = triangle[0].position.xy().as_dvec2();
        let v1 = triangle[1].position.xy().as_dvec2();
        let v2 = triangle[2].position.xy().as_dvec2();
        let inv_area = inv_area as f64;

        let w0_base = (v1.x * v2.y - v1.y * v2.x) * inv_area;
        let w1_base = (v2.x * v0.y - v2.y * v0.x) * inv_area;
        let w2_base = (v0.x * v1.y - v0.y * v1.x) * inv_area;

        let w0_x_component = (v1.y - v2.y) * inv_area;
        let w1_x_component = (v2.y - v0.y) * inv_area;
        let w2_x_component = (v0.y - v1.y) * inv_area;

        let w0_y_component = (v2.x - v1.x) * inv_area;
        let w1_y_component = (v0.x - v2.x) * inv_area;
        let w2_y_component = (v1.x - v0.x) * inv_area;

        let w0 = StaticDataComponent {
            base: w0_base,
            x_component: w0_x_component,
            y_component: w0_y_component,
        };
        let w1 = StaticDataComponent {
            base: w1_base,
            x_component: w1_x_component,
            y_component: w1_y_component,
        };
        let w2 = StaticDataComponent {
            base: w2_base,
            x_component: w2_x_component,
            y_component: w2_y_component,
        };

        Self(w0, w1, w2)
    }

    fn barycentric_computations(&self, y: f32) -> BarycentricComputations {
        BarycentricComputations(
            self.0.barycentric_computation(y),
            self.1.barycentric_computation(y),
            self.2.barycentric_computation(y),
        )
    }
}

#[derive(Copy, Clone, Default)]
struct BarycentricComputation {
    value: f64,
    x_delta: f64,
}

impl BarycentricComputation {
    fn barycentric_coordinate(&self, x: f32) -> f32 {
        (self.value + self.x_delta * x as f64) as f32
    }
}

#[derive(Copy, Clone, Default)]
struct BarycentricComputations(
    BarycentricComputation,
    BarycentricComputation,
    BarycentricComputation,
);

impl BarycentricComputations {
    fn barycentric_coordinates(&self, x: f32) -> BarycentricCoordinates {
        BarycentricCoordinates(
            self.0.barycentric_coordinate(x),
            self.1.barycentric_coordinate(x),
            self.2.barycentric_coordinate(x),
        )
    }
}

#[derive(Copy, Clone)]
struct BarycentricCoordinates(f32, f32, f32);

impl BarycentricCoordinates {
    fn interpolate<T>(&self, v0: T, v1: T, v2: T) -> T
    where
        T: Mul<f32, Output = T> + Add<Output = T>,
    {
        v0 * self.0 + v1 * self.1 + v2 * self.2
    }

    fn interpolate_arr<T, const N: usize>(&self, a0: &[T; N], a1: &[T; N], a2: &[T; N]) -> [T; N]
    where
        T: Mul<f32, Output = T> + Add<Output = T> + Copy + Default,
    {
        let mut res_arr = [Default::default(); N];
        for ((&v0_elem, &v1_elem, &v2_elem), res_elem) in a0
            .iter()
            .zip(a1)
            .zip(a2)
            .map(|v012_elements| (v012_elements.0 .0, v012_elements.0 .1, v012_elements.1))
            .zip(&mut res_arr)
        {
            *res_elem = v0_elem * self.0 + v1_elem * self.1 + v2_elem * self.2;
        }
        res_arr
    }

    fn are_in_triangle(&self) -> bool {
        self.0.to_bits() >> 31 == self.1.to_bits() >> 31
            && self.1.to_bits() >> 31 == self.2.to_bits() >> 31
    }
}

fn calculate_double_area<const A: usize>(triangle: &[UnshadedFragment<A>; 3]) -> f32 {
    let a = triangle[0].position.xy() - triangle[1].position.xy();
    let b = triangle[2].position.xy() - triangle[1].position.xy();
    a.x * b.y - a.y * b.x
}

fn skip_triangle(area: f32, cull_face: Face) -> bool {
    if area > 0.0 {
        if cull_face == Face::Cw {
            return true;
        }
    } else {
        if cull_face == Face::Ccw {
            return true;
        }
    }
    return false;
}

#[derive(Copy, Clone, Debug)]
struct PassedSamples<const S: usize> {
    passed_samples: [u32; S],
    passed_samples_count: usize,
    center_offset: glam::Vec2,
}

impl<const S: usize> PassedSamples<S> {
    unsafe fn push(&mut self, sample: u32) {
        *self
            .passed_samples
            .get_unchecked_mut(self.passed_samples_count) = sample;
        self.passed_samples_count += 1;
    }

    unsafe fn get_passed_sample_at(&self, index: usize) -> u32 {
        *self.passed_samples.get_unchecked(index)
    }

    fn add_offset(&mut self, offset: glam::Vec2) {
        self.center_offset += offset;
    }

    fn center_offset(&self) -> glam::Vec2 {
        self.center_offset / self.passed_samples_count as f32
    }

    fn count(&self) -> usize {
        self.passed_samples_count
    }
}

impl<const S: usize> Default for PassedSamples<S> {
    fn default() -> Self {
        Self {
            passed_samples: [0; S],
            passed_samples_count: 0,
            center_offset: glam::Vec2::ZERO,
        }
    }
}

fn process_bounds_tiled<
    'a,
    'b: 'a,
    U: Copy + Send,
    T: Copy + Default + Send,
    FS: Fn(UnshadedFragment<A>, U) -> T + Sync,
    B: Fn(&T, &T) -> T + Sync,
    MS: Multisampler<S> + Sync,
    const A: usize,
    const S: usize,
    const TR: usize,
    const TS: usize,
>(
    scope: &Scope<'a>,
    bounds: Bounds,
    triangles: &'b [[UnshadedFragment<A>; 3]; TR],
    fragment_tools: &'b UnsafeFragmentTools<U, T, FS, B, A, S>,
    depth_tools: &'b UnsafeDepthTools<S>,
    multisampler: &'b MS,
    static_data: &'b [StaticData; TR],
    lines: &'b [Line; 3],
) {
    let mut barycentric_computations = [[[BarycentricComputations::default(); TR]; S]; TS];
    for (y, barycentric_computations) in
        (bounds.min.y..bounds.max.y).zip(&mut barycentric_computations)
    {
        *barycentric_computations = multisampler.y_offsets().map(|y_offset| {
            static_data.map(|static_data| static_data.barycentric_computations(y as f32 + y_offset))
        });
    }

    let x_max_aligned = bounds.max.x - (bounds.max.x - bounds.min.x) % TS as u32;
    for x_min in (bounds.min.x..x_max_aligned).step_by(TS) {
        let tile_bounds = Bounds {
            min: glam::uvec2(x_min, bounds.min.y),
            max: glam::uvec2(x_min + TS as u32, bounds.max.y),
        };

        check_and_process(
            scope,
            tile_bounds,
            &triangles,
            fragment_tools,
            depth_tools,
            multisampler,
            static_data,
            barycentric_computations,
            lines,
        )
    }

    let tile_bounds = Bounds {
        min: glam::uvec2(x_max_aligned, bounds.min.y),
        max: glam::uvec2(bounds.max.x, bounds.max.y),
    };

    check_and_process(
        scope,
        tile_bounds,
        &triangles,
        fragment_tools,
        depth_tools,
        multisampler,
        static_data,
        barycentric_computations,
        lines,
    )
}

fn check_and_process<
    'a,
    'b: 'a,
    U: Copy + Send,
    T: Copy + Default + Send,
    FS: Fn(UnshadedFragment<A>, U) -> T + Sync,
    B: Fn(&T, &T) -> T + Sync,
    MS: Multisampler<S> + Sync,
    const A: usize,
    const S: usize,
    const TR: usize,
    const TS: usize,
>(
    scope: &Scope<'a>,
    bounds: Bounds,
    triangles: &'b [[UnshadedFragment<A>; 3]; TR],
    fragment_tools: &'b UnsafeFragmentTools<U, T, FS, B, A, S>,
    depth_tools: &'b UnsafeDepthTools<S>,
    multisampler: &'b MS,
    static_data: &'b [StaticData; TR],
    barycentric_computations: [[[BarycentricComputations; TR]; S]; TS],
    lines: &'b [Line; 3],
) {
    if triangles
        .iter()
        .any(|triangle| bounds.intersect_or_contain(&lines, &triangle))
    {
        scope.spawn(move |_| {
            process_bounds_with_samples(
                &bounds,
                &triangles,
                fragment_tools,
                depth_tools,
                multisampler,
                static_data,
                &barycentric_computations,
                false,
            );
        });
    } else {
        let middle = bounds.middle();

        let barycentric_coordinates = static_data.map(|static_data| {
            static_data
                .barycentric_computations(middle.y)
                .barycentric_coordinates(middle.x)
        });

        if barycentric_coordinates
            .iter()
            .any(|barycentric_coordinates| barycentric_coordinates.are_in_triangle())
        {
            scope.spawn(move |_| {
                process_bounds_with_samples(
                    &bounds,
                    &triangles,
                    fragment_tools,
                    depth_tools,
                    multisampler,
                    static_data,
                    &barycentric_computations,
                    true,
                );
            });
        }
    }
}

fn process_bounds<U, T, FS, B, MS, const A: usize, const S: usize, const TR: usize>(
    bounds: &Bounds,
    triangles: &[[UnshadedFragment<A>; 3]; TR],
    fragment_tools: &UnsafeFragmentTools<U, T, FS, B, A, S>,
    depth_tools: &UnsafeDepthTools<S>,
    multisampler: &MS,
    static_data: &[StaticData; TR],
    bounds_covered: bool,
) where
    U: Copy + Send,
    T: Copy + Default + Send,
    FS: Fn(UnshadedFragment<A>, U) -> T + Sync,
    B: Fn(&T, &T) -> T + Sync,
    MS: Multisampler<S>,
{
    for y in bounds.min.y..bounds.max.y {
        let barycentric_computations = multisampler.y_offsets().map(|y_offset| {
            static_data.map(|static_data| static_data.barycentric_computations(y as f32 + y_offset))
        });

        process_row(
            bounds.min.x..bounds.max.x,
            triangles,
            fragment_tools,
            depth_tools,
            multisampler,
            static_data,
            &barycentric_computations,
            bounds_covered,
            y,
        );
    }
}

fn process_bounds_with_samples<
    U: Copy + Send,
    T: Copy + Default + Send,
    FS: Fn(UnshadedFragment<A>, U) -> T + Sync,
    B: Fn(&T, &T) -> T + Sync,
    MS: Multisampler<S>,
    const A: usize,
    const S: usize,
    const TR: usize,
    const TS: usize,
>(
    bounds: &Bounds,
    triangles: &[[UnshadedFragment<A>; 3]; TR],
    fragment_tools: &UnsafeFragmentTools<U, T, FS, B, A, S>,
    depth_tools: &UnsafeDepthTools<S>,
    multisampler: &MS,
    static_data: &[StaticData; TR],
    barycentric_computations: &[[[BarycentricComputations; TR]; S]; TS],
    bounds_covered: bool,
) {
    for (y, barycentric_computations) in (bounds.min.y..bounds.max.y).zip(barycentric_computations)
    {
        process_row(
            bounds.min.x..bounds.max.x,
            triangles,
            fragment_tools,
            depth_tools,
            multisampler,
            static_data,
            barycentric_computations,
            bounds_covered,
            y,
        );
    }
}

fn process_row<U, T, FS, B, MS, const A: usize, const S: usize, const TR: usize>(
    x_range: Range<u32>,
    triangles: &[[UnshadedFragment<A>; 3]; TR],
    fragment_tools: &UnsafeFragmentTools<U, T, FS, B, A, S>,
    depth_tools: &UnsafeDepthTools<S>,
    multisampler: &MS,
    static_data: &[StaticData; TR],
    barycentric_computations: &[[BarycentricComputations; TR]; S],
    row_covered: bool,
    y: u32,
) where
    U: Copy + Send,
    T: Copy + Default + Send,
    FS: Fn(UnshadedFragment<A>, U) -> T + Sync,
    B: Fn(&T, &T) -> T + Sync,
    MS: Multisampler<S>,
{
    for x in x_range {
        let ref x_offsets = multisampler.x_offsets(x, y);

        let mut passed_samples = PassedSamples::<S>::default();

        for (sample, ((&x_offset, &y_offset), barycentric_computations)) in x_offsets
            .iter()
            .zip(multisampler.y_offsets())
            .zip(barycentric_computations)
            .enumerate()
        {
            let sample = sample as u32;

            for (barycentric_computations, triangle) in
                barycentric_computations.iter().zip(triangles)
            {
                let barycentric_coordinates =
                    barycentric_computations.barycentric_coordinates(x as f32 + x_offset);

                if row_covered || barycentric_coordinates.are_in_triangle() {
                    let depth = barycentric_coordinates.interpolate(
                        triangle[0].position.z,
                        triangle[1].position.z,
                        triangle[2].position.z,
                    );
                    unsafe {
                        if let Some(ref mut depth_tools) = *depth_tools.0 {
                            let depth_test_passed = depth_tools.compare(x, y, sample, depth);

                            if depth_test_passed {
                                passed_samples.push(sample);
                                passed_samples.add_offset(glam::vec2(x_offset, y_offset));
                                if depth_tools.depth_state.write_depth {
                                    depth_tools.write_depth(x, y, sample, depth);
                                }
                                break;
                            }
                        } else {
                            passed_samples.push(sample);
                            passed_samples.add_offset(glam::vec2(x_offset, y_offset));
                            break;
                        }
                    }
                }
            }
        }

        if passed_samples.count() != 0 {
            let static_data = static_data[0];
            let triangle = triangles[0];

            let xy = glam::vec2(x as f32, y as f32) + passed_samples.center_offset();
            let barycentric_coordinates = static_data
                .barycentric_computations(xy.y)
                .barycentric_coordinates(xy.x);

            let zw = barycentric_coordinates.interpolate(
                triangle[0].position.zw(),
                triangle[1].position.zw(),
                triangle[2].position.zw(),
            );
            let mut position: glam::Vec4 = (xy, zw).into();
            position.w = 1.0 / position.w;
            let attributes = barycentric_coordinates
                .interpolate_arr(
                    &triangle[0].attributes,
                    &triangle[1].attributes,
                    &triangle[2].attributes,
                )
                .map(|attr| attr * position.w);
            let fragment_value = unsafe {
                (*fragment_tools.0).shade_fragment(UnshadedFragment {
                    position,
                    attributes,
                })
            };

            for i in 0..passed_samples.count() {
                unsafe {
                    let sample = passed_samples.get_passed_sample_at(i);
                    (*fragment_tools.0).blend_and_write(x, y, sample, &fragment_value);
                }
            }
        }
    }
}
