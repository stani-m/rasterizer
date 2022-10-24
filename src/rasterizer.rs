use std::mem;
use std::num::NonZeroU32;
use std::ops::{Add, Mul};

use glam::Vec4Swizzles;
use rayon_core::{ThreadPool, ThreadPoolBuilder};

use crate::{Buffer, DepthState, FragmentInput, Multisampler, PixelCenterOffset};

pub trait Rasterizer<const V: usize, const A: usize, const S: usize> {
    fn set_screen_size(&mut self, width: u32, height: u32);
    fn rasterize<'a, 'b, U, T, DF, FS, B, MS>(
        &self,
        shape: [FragmentInput<A>; V],
        fragment_tools: &mut FragmentTools<'a, U, T, FS, B, A, S>,
        depth_tools: &mut Option<DepthTools<'b, DF, S>>,
        multisampler: &MS,
    ) where
        U: Copy + Send,
        T: Copy + Default + Send,
        DF: Fn(f32, f32) -> bool + Sync,
        FS: Fn(FragmentInput<A>, U) -> T + Sync,
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
    FS: Fn(FragmentInput<A>, U) -> T,
    B: Fn(&T, &T) -> T,
{
    fn shade_fragment(&self, fragment_input: FragmentInput<A>) -> T {
        (self.fragment_shader)(fragment_input, self.uniforms)
    }

    fn blend_and_write(&mut self, x: u32, y: u32, sample: u32, value: &T) {
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

pub struct DepthTools<'a, DF, const S: usize> {
    pub(crate) depth_state: &'a DepthState<DF>,
    pub(crate) depth_buffer: &'a mut Buffer<f32, S>,
}

impl<'a, DF: Fn(f32, f32) -> bool, const S: usize> DepthTools<'a, DF, S> {
    fn compare(&self, x: u32, y: u32, sample: u32, depth: f32) -> bool {
        let dst_depth = self.depth_buffer[[x, y, sample]];
        (self.depth_state.depth_function)(depth, dst_depth)
    }

    fn write(&mut self, x: u32, y: u32, sample: u32, depth: f32) {
        self.depth_buffer[[x, y, sample]] = depth;
    }
}

struct UnsafeDepthTools<'a, DF: Sync, const S: usize>(*mut Option<DepthTools<'a, DF, S>>);
unsafe impl<'a, DF: Sync, const S: usize> Send for UnsafeDepthTools<'a, DF, S> {}
unsafe impl<'a, DF: Sync, const S: usize> Sync for UnsafeDepthTools<'a, DF, S> {}

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
    fn set_screen_size(&mut self, width: u32, height: u32) {
        self.width = width as i32;
        self.height = height as i32;
    }

    fn rasterize<U, T, DF, FS, B, MS>(
        &self,
        shape: [FragmentInput<A>; 2],
        fragment_tools: &mut FragmentTools<U, T, FS, B, A, S>,
        depth_tools: &mut Option<DepthTools<DF, S>>,
        _: &MS,
    ) where
        U: Copy,
        T: Copy + Default,
        DF: Fn(f32, f32) -> bool,
        FS: Fn(FragmentInput<A>, U) -> T,
        B: Fn(&T, &T) -> T,
    {
        let [mut from, mut to] = shape;
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
                    let fragment_value = fragment_tools.shade_fragment(FragmentInput {
                        position: glam::vec4(x as f32, y as f32, z, zw[1]),
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
                                    depth_tools.write(x as u32, y as u32, sample, z);
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
                        x += adjust;
                        threshold += threshold_inc;
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
    fn new(from: &FragmentInput<A>, to: &FragmentInput<A>, steps: i32) -> Self {
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
    width: f32,
    rasterizer: EdgeFunctionTiledRasterizer,
}

impl AALineRasterizer {
    pub fn new(width: f32) -> Self {
        Self {
            width,
            rasterizer: EdgeFunctionTiledRasterizer::new(EdgeFunctionTiledRasterizerDescriptor {
                cull_face: None,
                tile_size: NonZeroU32::new(32).unwrap(),
                thread_count: 0,
            }),
        }
    }
}

impl<const A: usize, const S: usize> Rasterizer<2, A, S> for AALineRasterizer {
    fn set_screen_size(&mut self, width: u32, height: u32) {
        self.rasterizer.screen_bounds.max = glam::uvec2(width, height);
    }

    fn rasterize<'a, 'b, U, T, DF, FS, B, MS>(
        &self,
        shape: [FragmentInput<A>; 2],
        fragment_tools: &mut FragmentTools<'a, U, T, FS, B, A, S>,
        depth_tools: &mut Option<DepthTools<'b, DF, S>>,
        multisampler: &MS,
    ) where
        U: Copy + Send,
        T: Copy + Default + Send,
        DF: Fn(f32, f32) -> bool + Sync,
        FS: Fn(FragmentInput<A>, U) -> T + Sync,
        B: Fn(&T, &T) -> T + Sync,
        MS: Multisampler<S> + Sync,
    {
        let vector = (shape[1].position.xy() - shape[0].position.xy()).normalize();
        let normal = glam::vec4(-vector.y, vector.x, 0.0, 0.0) * self.width * 0.5;

        let a0 = FragmentInput {
            position: shape[0].position + normal,
            ..shape[0]
        };
        let a1 = FragmentInput {
            position: shape[0].position - normal,
            ..shape[0]
        };
        let b0 = FragmentInput {
            position: shape[1].position + normal,
            ..shape[1]
        };
        let b1 = FragmentInput {
            position: shape[1].position - normal,
            ..shape[1]
        };

        self.rasterizer
            .rasterize([a0, a1, b0], fragment_tools, depth_tools, multisampler);
        self.rasterizer
            .rasterize([a1, b0, b1], fragment_tools, depth_tools, multisampler);
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Face {
    Cw,
    Ccw,
}

#[derive(Copy, Clone, Debug)]
pub struct EdgeFunctionRasterizer {
    screen_bounds: Bounds,
    cull_face: Option<Face>,
}

impl EdgeFunctionRasterizer {
    pub fn new(cull_face: Option<Face>) -> Self {
        Self {
            screen_bounds: Bounds {
                min: glam::uvec2(0, 0),
                max: glam::uvec2(0, 0),
            },
            cull_face,
        }
    }
}

impl<const A: usize, const S: usize> Rasterizer<3, A, S> for EdgeFunctionRasterizer {
    fn set_screen_size(&mut self, width: u32, height: u32) {
        self.screen_bounds.max = glam::uvec2(width, height);
    }

    fn rasterize<'a, 'b, U, T, DF, FS, B, MS>(
        &self,
        shape: [FragmentInput<A>; 3],
        fragment_tools: &mut FragmentTools<'a, U, T, FS, B, A, S>,
        depth_tools: &mut Option<DepthTools<'b, DF, S>>,
        multisampler: &MS,
    ) where
        U: Copy + Send,
        T: Copy + Default + Send,
        DF: Fn(f32, f32) -> bool + Sync,
        FS: Fn(FragmentInput<A>, U) -> T + Sync,
        B: Fn(&T, &T) -> T + Sync,
        MS: Multisampler<S> + Sync,
    {
        let area = calculate_area(&shape);

        if skip_triangle(area, self.cull_face) {
            return;
        }

        let inv_area = -1.0 / area;

        let bounds = Bounds::from(&shape).clamp(self.screen_bounds);

        let (w0, w1, w2) = prepare_static_data(&shape, inv_area);

        let mut samples = [SampleData {
            w0: BarycentricCoordinate::new(&w0),
            w1: BarycentricCoordinate::new(&w1),
            w2: BarycentricCoordinate::new(&w2),
        }; S];

        // DepthTools and FragmentTools don't need to be unsafe here but it makes code reuse much easier
        let fragment_tools = &UnsafeFragmentTools(fragment_tools as *mut _);
        let depth_tools = &UnsafeDepthTools(depth_tools as *mut _);

        process_tile(
            &bounds,
            &shape,
            fragment_tools,
            depth_tools,
            multisampler,
            &mut samples,
            false,
        );
    }
}

#[derive(Debug)]
pub struct EdgeFunctionMTRasterizer {
    screen_bounds: Bounds,
    cull_face: Option<Face>,
    thread_pool: ThreadPool,
}

impl EdgeFunctionMTRasterizer {
    pub fn new(cull_face: Option<Face>, thread_count: usize) -> Self {
        Self {
            screen_bounds: Bounds {
                min: glam::uvec2(0, 0),
                max: glam::uvec2(0, 0),
            },
            cull_face,
            thread_pool: ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .unwrap(),
        }
    }
}

impl<const A: usize, const S: usize> Rasterizer<3, A, S> for EdgeFunctionMTRasterizer {
    fn set_screen_size(&mut self, width: u32, height: u32) {
        self.screen_bounds.max = glam::uvec2(width, height);
    }

    fn rasterize<'a, 'b, U, T, DF, FS, B, MS>(
        &self,
        shape: [FragmentInput<A>; 3],
        fragment_tools: &mut FragmentTools<'a, U, T, FS, B, A, S>,
        depth_tools: &mut Option<DepthTools<'b, DF, S>>,
        multisampler: &MS,
    ) where
        U: Copy + Send,
        T: Copy + Default + Send,
        DF: Fn(f32, f32) -> bool + Sync,
        FS: Fn(FragmentInput<A>, U) -> T + Sync,
        B: Fn(&T, &T) -> T + Sync,
        MS: Multisampler<S> + Sync,
    {
        let area = calculate_area(&shape);

        if skip_triangle(area, self.cull_face) {
            return;
        }

        let inv_area = -1.0 / area;

        let bounds = Bounds::from(&shape).clamp(self.screen_bounds);

        let (w0, w1, w2) = prepare_static_data(&shape, inv_area);

        let mut samples = [SampleData {
            w0: BarycentricCoordinate::new(&w0),
            w1: BarycentricCoordinate::new(&w1),
            w2: BarycentricCoordinate::new(&w2),
        }; S];

        let fragment_tools = &UnsafeFragmentTools(fragment_tools as *mut _);
        let depth_tools = &UnsafeDepthTools(depth_tools as *mut _);

        self.thread_pool.in_place_scope(|scope| {
            for y in bounds.min.y..bounds.max.y {
                scope.spawn(move |_| {
                    process_row(
                        &bounds,
                        &shape,
                        fragment_tools,
                        depth_tools,
                        multisampler,
                        &mut samples,
                        false,
                        y,
                    );
                });
            }
        });
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EdgeFunctionTiledRasterizerDescriptor {
    pub cull_face: Option<Face>,
    pub tile_size: NonZeroU32,
    pub thread_count: u32,
}

#[derive(Debug)]
pub struct EdgeFunctionTiledRasterizer {
    screen_bounds: Bounds,
    cull_face: Option<Face>,
    tile_size: NonZeroU32,
    thread_pool: ThreadPool,
}

impl EdgeFunctionTiledRasterizer {
    pub fn new(descriptor: EdgeFunctionTiledRasterizerDescriptor) -> Self {
        Self {
            screen_bounds: Bounds {
                min: glam::uvec2(0, 0),
                max: glam::uvec2(0, 0),
            },
            cull_face: descriptor.cull_face,
            tile_size: descriptor.tile_size,
            thread_pool: ThreadPoolBuilder::new()
                .num_threads(descriptor.thread_count as usize)
                .build()
                .unwrap(),
        }
    }

    fn check_and_process<U, T, DF, FS, B, MS, const A: usize, const S: usize>(
        bounds: &Bounds,
        shape: &[FragmentInput<A>; 3],
        fragment_tools: &UnsafeFragmentTools<U, T, FS, B, A, S>,
        depth_tools: &UnsafeDepthTools<DF, S>,
        multisampler: &MS,
        mut samples: [SampleData; S],
        lines: &[Line; 3],
        points: &[glam::Vec2; 3],
    ) where
        U: Copy + Send,
        T: Copy + Default + Send,
        DF: Fn(f32, f32) -> bool + Sync,
        FS: Fn(FragmentInput<A>, U) -> T + Sync,
        B: Fn(&T, &T) -> T + Sync,
        MS: Multisampler<S>,
    {
        if bounds.intersect_or_contain(&lines, &points) {
            process_tile(
                &bounds,
                &shape,
                fragment_tools,
                depth_tools,
                multisampler,
                &mut samples,
                false,
            );
        } else {
            let mut sample = samples[0];

            let middle = bounds.middle();
            sample.compute_row_bary_coords(middle.y);
            sample.compute_barycentric_coords(middle.x);

            if sample.is_in_triangle() {
                process_tile(
                    &bounds,
                    &shape,
                    fragment_tools,
                    depth_tools,
                    multisampler,
                    &mut samples,
                    true,
                );
            }
        }
    }
}

impl<const A: usize, const S: usize> Rasterizer<3, A, S> for EdgeFunctionTiledRasterizer {
    fn set_screen_size(&mut self, width: u32, height: u32) {
        self.screen_bounds.max = glam::uvec2(width, height);
    }

    fn rasterize<'a, 'b, U, T, DF, FS, B, MS>(
        &self,
        shape: [FragmentInput<A>; 3],
        fragment_tools: &mut FragmentTools<'a, U, T, FS, B, A, S>,
        depth_tools: &mut Option<DepthTools<'b, DF, S>>,
        multisampler: &MS,
    ) where
        U: Copy + Send,
        T: Copy + Default + Send,
        DF: Fn(f32, f32) -> bool + Sync,
        FS: Fn(FragmentInput<A>, U) -> T + Sync,
        B: Fn(&T, &T) -> T + Sync,
        MS: Multisampler<S> + Sync,
    {
        let area = calculate_area(&shape);

        if skip_triangle(area, self.cull_face) {
            return;
        }

        let inv_area = -1.0 / area;

        let bounds = Bounds::from(&shape).clamp(self.screen_bounds);
        let max_tile = bounds.max - (bounds.max - bounds.min) % self.tile_size.get();

        let (w0, w1, w2) = prepare_static_data(&shape, inv_area);

        let samples = [SampleData {
            w0: BarycentricCoordinate::new(&w0),
            w1: BarycentricCoordinate::new(&w1),
            w2: BarycentricCoordinate::new(&w2),
        }; S];

        let triangle_lines = [
            Line::new(shape[0].position.xy(), shape[1].position.xy()),
            Line::new(shape[1].position.xy(), shape[2].position.xy()),
            Line::new(shape[2].position.xy(), shape[0].position.xy()),
        ];
        let triangle_points = [
            shape[0].position.xy(),
            shape[1].position.xy(),
            shape[2].position.xy(),
        ];

        let fragment_tools = &UnsafeFragmentTools(fragment_tools as *mut _);
        let depth_tools = &UnsafeDepthTools(depth_tools as *mut _);

        self.thread_pool.in_place_scope(|scope| {
            for y_min in (bounds.min.y..max_tile.y).step_by(self.tile_size.get() as usize) {
                let y_max = y_min + self.tile_size.get();
                for x_min in (bounds.min.x..max_tile.x).step_by(self.tile_size.get() as usize) {
                    let tile_bounds = Bounds {
                        min: glam::uvec2(x_min, y_min),
                        max: glam::uvec2(x_min + self.tile_size.get(), y_max),
                    };

                    scope.spawn(move |_| {
                        Self::check_and_process(
                            &tile_bounds,
                            &shape,
                            fragment_tools,
                            depth_tools,
                            multisampler,
                            samples,
                            &triangle_lines,
                            &triangle_points,
                        )
                    });
                }

                let tile_bounds = Bounds {
                    min: glam::uvec2(max_tile.x, y_min),
                    max: glam::uvec2(bounds.max.x, y_max),
                };

                scope.spawn(move |_| {
                    Self::check_and_process(
                        &tile_bounds,
                        &shape,
                        fragment_tools,
                        depth_tools,
                        multisampler,
                        samples,
                        &triangle_lines,
                        &triangle_points,
                    )
                });
            }

            for x_min in (bounds.min.x..max_tile.x).step_by(self.tile_size.get() as usize) {
                let tile_bounds = Bounds {
                    min: glam::uvec2(x_min, max_tile.y),
                    max: glam::uvec2(x_min + self.tile_size.get(), bounds.max.y),
                };

                scope.spawn(move |_| {
                    Self::check_and_process(
                        &tile_bounds,
                        &shape,
                        fragment_tools,
                        depth_tools,
                        multisampler,
                        samples,
                        &triangle_lines,
                        &triangle_points,
                    )
                });
            }

            let tile_bounds = Bounds {
                min: glam::uvec2(max_tile.x, max_tile.y),
                max: glam::uvec2(bounds.max.x, bounds.max.y),
            };

            scope.spawn(move |_| {
                Self::check_and_process(
                    &tile_bounds,
                    &shape,
                    fragment_tools,
                    depth_tools,
                    multisampler,
                    samples,
                    &triangle_lines,
                    &triangle_points,
                )
            });
        });
    }
}

#[derive(Copy, Clone, Debug)]
struct Bounds {
    min: glam::UVec2,
    max: glam::UVec2,
}

impl Bounds {
    fn intersect_or_contain(&self, lines: &[Line; 3], points: &[glam::Vec2; 3]) -> bool {
        let min_bound = self.min.as_vec2();
        let max_bound = self.max.as_vec2();
        let x_range = min_bound.x..=max_bound.x;
        let y_range = min_bound.y..=max_bound.y;
        let mut contain = true;
        for (line, point) in lines.into_iter().zip(points.into_iter()) {
            if !x_range.contains(&point.x) || !y_range.contains(&point.y) {
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

    fn clamp(self, other: Self) -> Self {
        Bounds {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        }
    }
}

impl<const V: usize, const A: usize> From<&[FragmentInput<A>; V]> for Bounds {
    fn from(shape: &[FragmentInput<A>; V]) -> Self {
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
}

impl Line {
    fn new(a: glam::Vec2, b: glam::Vec2) -> Self {
        let slope = (b.y - a.y) / (b.x - a.x);
        Self {
            slope,
            inv_slope: 1.0 / slope,
            intercept: a.y - slope * a.x,
        }
    }

    fn solve_y(&self, x: f32) -> f32 {
        self.slope * x + self.intercept
    }

    fn solve_x(&self, y: f32) -> f32 {
        (y - self.intercept) * self.inv_slope
    }
}

#[derive(Copy, Clone)]
struct StaticData {
    base: f32,
    x_delta: f32,
    y_delta: f32,
}

#[derive(Copy, Clone)]
struct BarycentricCoordinate<'a> {
    static_data: &'a StaticData,
    row_value: f32,
    value: f32,
}

impl<'a> BarycentricCoordinate<'a> {
    fn new(static_data: &'a StaticData) -> Self {
        Self {
            static_data,
            row_value: 0.0,
            value: 0.0,
        }
    }

    fn compute_row(&mut self, y: f32) {
        self.row_value = self.static_data.y_delta * y + self.static_data.base;
    }

    fn compute_value(&mut self, x: f32) {
        self.value = self.static_data.x_delta * x + self.row_value;
    }
}

#[derive(Copy, Clone)]
struct SampleData<'a> {
    w0: BarycentricCoordinate<'a>,
    w1: BarycentricCoordinate<'a>,
    w2: BarycentricCoordinate<'a>,
}

impl<'a> SampleData<'a> {
    fn compute_row_bary_coords(&mut self, y: f32) {
        self.w0.compute_row(y);
        self.w1.compute_row(y);
        self.w2.compute_row(y);
    }

    fn compute_barycentric_coords(&mut self, x: f32) {
        self.w0.compute_value(x);
        self.w1.compute_value(x);
        self.w2.compute_value(x);
    }

    fn interpolate<T>(&self, v0: T, v1: T, v2: T) -> T
    where
        T: Mul<f32, Output = T> + Add<Output = T>,
    {
        v0 * self.w0.value + v1 * self.w1.value + v2 * self.w2.value
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
            *res_elem = v0_elem * self.w0.value + v1_elem * self.w1.value + v2_elem * self.w2.value;
        }
        res_arr
    }

    fn is_in_triangle(&self) -> bool {
        self.w0.value.to_bits() >> 31 == self.w1.value.to_bits() >> 31
            && self.w1.value.to_bits() >> 31 == self.w2.value.to_bits() >> 31
    }
}

fn calculate_area<const A: usize>(shape: &[FragmentInput<A>; 3]) -> f32 {
    let a = shape[0].position.xy() - shape[1].position.xy();
    let b = shape[2].position.xy() - shape[1].position.xy();
    a.x * b.y - a.y * b.x
}

fn skip_triangle(area: f32, cull_face: Option<Face>) -> bool {
    if area > 0.0 {
        if cull_face == Some(Face::Cw) {
            return true;
        }
    } else {
        if cull_face == Some(Face::Ccw) {
            return true;
        }
    }
    return false;
}

fn prepare_static_data<const A: usize>(
    shape: &[FragmentInput<A>; 3],
    inv_area: f32,
) -> (StaticData, StaticData, StaticData) {
    let w0_base = (shape[1].position.x * shape[2].position.y
        - shape[1].position.y * shape[2].position.x)
        * inv_area;
    let w1_base = (shape[2].position.x * shape[0].position.y
        - shape[2].position.y * shape[0].position.x)
        * inv_area;
    let w2_base = (shape[0].position.x * shape[1].position.y
        - shape[0].position.y * shape[1].position.x)
        * inv_area;

    let w0_x_delta = (shape[1].position.y - shape[2].position.y) * inv_area;
    let w1_x_delta = (shape[2].position.y - shape[0].position.y) * inv_area;
    let w2_x_delta = (shape[0].position.y - shape[1].position.y) * inv_area;

    let w0_y_delta = (shape[2].position.x - shape[1].position.x) * inv_area;
    let w1_y_delta = (shape[0].position.x - shape[2].position.x) * inv_area;
    let w2_y_delta = (shape[1].position.x - shape[0].position.x) * inv_area;

    let w0 = StaticData {
        base: w0_base,
        x_delta: w0_x_delta,
        y_delta: w0_y_delta,
    };
    let w1 = StaticData {
        base: w1_base,
        x_delta: w1_x_delta,
        y_delta: w1_y_delta,
    };
    let w2 = StaticData {
        base: w2_base,
        x_delta: w2_x_delta,
        y_delta: w2_y_delta,
    };

    (w0, w1, w2)
}

fn process_tile<U, T, DF, FS, B, MS, const A: usize, const S: usize>(
    bounds: &Bounds,
    shape: &[FragmentInput<A>; 3],
    fragment_tools: &UnsafeFragmentTools<U, T, FS, B, A, S>,
    depth_tools: &UnsafeDepthTools<DF, S>,
    multisampler: &MS,
    samples: &mut [SampleData; S],
    tile_covered: bool,
) where
    U: Copy + Send,
    T: Copy + Default + Send,
    DF: Fn(f32, f32) -> bool + Sync,
    FS: Fn(FragmentInput<A>, U) -> T + Sync,
    B: Fn(&T, &T) -> T + Sync,
    MS: Multisampler<S>,
{
    for y in bounds.min.y..bounds.max.y {
        process_row(
            bounds,
            shape,
            fragment_tools,
            depth_tools,
            multisampler,
            samples,
            tile_covered,
            y,
        );
    }
}

fn process_row<U, T, DF, FS, B, MS, const A: usize, const S: usize>(
    bounds: &Bounds,
    shape: &[FragmentInput<A>; 3],
    fragment_tools: &UnsafeFragmentTools<U, T, FS, B, A, S>,
    depth_tools: &UnsafeDepthTools<DF, S>,
    multisampler: &MS,
    samples: &mut [SampleData; S],
    tile_covered: bool,
    y: u32,
) where
    U: Copy + Send,
    T: Copy + Default + Send,
    DF: Fn(f32, f32) -> bool + Sync,
    FS: Fn(FragmentInput<A>, U) -> T + Sync,
    B: Fn(&T, &T) -> T + Sync,
    MS: Multisampler<S>,
{
    let y_offsets = multisampler.y_offsets();

    for i in 0..S {
        samples[i].compute_row_bary_coords(y as f32 + y_offsets[i]);
    }

    for x in bounds.min.x..bounds.max.x {
        let ref x_offsets = multisampler.x_offsets(x, y);

        let mut covered_samples = [(0, 0.0); S];
        let mut covered_samples_len = 0;

        for i in 0..S {
            let sample = &mut samples[i];
            sample.compute_barycentric_coords(x as f32 + x_offsets[i]);

            if tile_covered || sample.is_in_triangle() {
                let depth = sample.interpolate(
                    shape[0].position.z,
                    shape[1].position.z,
                    shape[2].position.z,
                );
                if 0.0 < depth && depth < 1.0 {
                    let depth_test_passed = unsafe {
                        (*depth_tools.0).as_ref().map_or(true, |depth_tools| {
                            depth_tools.compare(x, y, i as u32, depth)
                        })
                    };
                    if depth_test_passed {
                        unsafe {
                            *covered_samples.get_unchecked_mut(covered_samples_len) =
                                (i as u32, depth);
                        }
                        covered_samples_len += 1;
                    }
                }
            }
        }

        if covered_samples_len != 0 {
            let (ref fragment_sample, xy) = match multisampler.center_offset() {
                PixelCenterOffset::Index(i) => {
                    let xy = glam::vec2(x as f32 + x_offsets[i], y as f32 + y_offsets[i]);
                    (samples[i], xy)
                }
                PixelCenterOffset::Offset(offset) => {
                    let mut fragment_sample = samples[0];
                    let xy = glam::vec2(x as f32, y as f32) + offset;
                    fragment_sample.compute_row_bary_coords(xy.y);
                    fragment_sample.compute_barycentric_coords(xy.x);
                    (fragment_sample, xy)
                }
            };
            let zw = fragment_sample.interpolate(
                shape[0].position.zw(),
                shape[1].position.zw(),
                shape[2].position.zw(),
            );
            let position = (xy, zw).into();
            let inv_w = 1.0 / zw[1];
            let attributes = fragment_sample
                .interpolate_arr(
                    &shape[0].attributes,
                    &shape[1].attributes,
                    &shape[2].attributes,
                )
                .map(|attr| attr * inv_w);
            let fragment_value = unsafe {
                (*fragment_tools.0).shade_fragment(FragmentInput {
                    position,
                    attributes,
                })
            };

            for i in 0..covered_samples_len {
                unsafe {
                    let (sample, depth) = *covered_samples.get_unchecked(i);
                    if let Some(ref mut depth_tools) = *depth_tools.0 {
                        if depth_tools.depth_state.write_depth {
                            depth_tools.write(x, y, sample, depth);
                        }
                    }
                    (*fragment_tools.0).blend_and_write(x, y, sample, &fragment_value);
                }
            }
        }
    }
}
