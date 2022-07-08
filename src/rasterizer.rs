use std::mem;
use std::ops::{Add, Mul};

use glam::Vec4Swizzles;

use crate::{Buffer, DepthState, FragmentInput, MultisampleState, PixelCenterOffset};

pub trait Rasterizer<const V: usize, const A: usize, const S: usize> {
    fn set_screen_size(&mut self, width: u32, height: u32);
    fn rasterize<'a, U, T, DF, FS, B>(
        &self,
        shape: [FragmentInput<A>; V],
        fragment_tools: &mut FragmentTools<'a, U, T, FS, B, A, S>,
        depth_tools: &mut Option<DepthTools<'a, DF, S>>,
        multisample_state: &MultisampleState<S>,
    ) where
        U: Copy,
        T: Copy + Default,
        DF: Fn(f32, f32) -> bool,
        FS: Fn(FragmentInput<A>, U) -> T,
        B: Fn(&T, &T) -> T;
}

pub struct FragmentTools<'a, U, T, FS, B, const A: usize, const S: usize>
where
    U: Copy,
    T: Copy + Default,
    FS: Fn(FragmentInput<A>, U) -> T,
    B: Fn(&T, &T) -> T,
{
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

pub struct DepthTools<'a, DF: Fn(f32, f32) -> bool, const S: usize> {
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

    fn rasterize<'a, U, T, DF, FS, B>(
        &self,
        shape: [FragmentInput<A>; 2],
        fragment_tools: &mut FragmentTools<'a, U, T, FS, B, A, S>,
        depth_tools: &mut Option<DepthTools<'a, DF, S>>,
        _: &MultisampleState<S>,
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
                    for sample in 0..(S as u32) {
                        if let Some(depth_tools) = depth_tools {
                            let depth_test_passed =
                                depth_tools.compare(x as u32, y as u32, sample, z);
                            if depth_test_passed {
                                let fragment_value = fragment_tools.shade_fragment(FragmentInput {
                                    position: glam::vec4(x as f32, y as f32, z, zw[1]),
                                    attributes,
                                });
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
                            let fragment_value = fragment_tools.shade_fragment(FragmentInput {
                                position: glam::vec4(x as f32, y as f32, z, zw[1]),
                                attributes,
                            });
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

#[derive(Copy, Clone, Debug)]
pub struct BresenhamTriangleRasterizer {
    width: i32,
    height: i32,
}

impl BresenhamTriangleRasterizer {
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
        }
    }
}

impl<const A: usize, const S: usize> Rasterizer<3, A, S> for BresenhamTriangleRasterizer {
    fn set_screen_size(&mut self, width: u32, height: u32) {
        self.width = width as i32;
        self.height = height as i32;
    }

    fn rasterize<'a, U, T, DF, FS, B>(
        &self,
        shape: [FragmentInput<A>; 3],
        fragment_tools: &mut FragmentTools<'a, U, T, FS, B, A, S>,
        depth_tools: &mut Option<DepthTools<'a, DF, S>>,
        _: &MultisampleState<S>,
    ) where
        U: Copy,
        T: Copy + Default,
        DF: Fn(f32, f32) -> bool,
        FS: Fn(FragmentInput<A>, U) -> T,
        B: Fn(&T, &T) -> T,
    {
        let width_range = 0..self.width;
        let height_range = 0..self.height;
        let depth_range = 0.0..=1.0;
        let mut check_and_perform_action =
            |x: i32, y: i32, zw: glam::Vec2, attributes: [f32; A]| {
                let z = zw[0];
                if width_range.contains(&x) && height_range.contains(&y) && depth_range.contains(&z)
                {
                    for sample in 0..(S as u32) {
                        if let Some(depth_tools) = depth_tools {
                            let depth_test_passed =
                                depth_tools.compare(x as u32, y as u32, sample, z);
                            if depth_test_passed {
                                let fragment_value = fragment_tools.shade_fragment(FragmentInput {
                                    position: glam::vec4(x as f32, y as f32, z, zw[1]),
                                    attributes,
                                });
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
                            let fragment_value = fragment_tools.shade_fragment(FragmentInput {
                                position: glam::vec4(x as f32, y as f32, z, zw[1]),
                                attributes,
                            });
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

        let mut top = &shape[0];
        let mut left = &shape[1];
        let mut right = &shape[2];

        if left.position.y > top.position.y {
            mem::swap(&mut top, &mut left);
        }
        if right.position.y > top.position.y {
            mem::swap(&mut top, &mut right);
        }
        if left.position.x < top.position.x {
            if right.position.x < top.position.x {
                let [left_run, left_rise] = (left.position.xy() - top.position.xy()).to_array();
                let left_slope = left_run / left_rise;
                let [right_run, right_rise] = (right.position.xy() - top.position.xy()).to_array();
                let right_slope = right_run / right_rise;
                if left_slope < right_slope {
                    mem::swap(&mut left, &mut right);
                }
            }
        }
        if left.position.x >= top.position.x {
            if right.position.x > top.position.x {
                let [left_run, left_rise] = (left.position.xy() - top.position.xy()).to_array();
                let left_slope = left_run / left_rise;
                let [right_run, right_rise] = (right.position.xy() - top.position.xy()).to_array();
                let right_slope = right_run / right_rise;
                if left_slope < right_slope {
                    mem::swap(&mut left, &mut right);
                }
            } else {
                mem::swap(&mut left, &mut right);
            }
        }

        struct BresenhamData {
            rise: i32,
            delta: i32,
            threshold: i32,
            threshold_inc: i32,
            adjust: i32,
            x: i32,
            offset: i32,
        }

        impl BresenhamData {
            fn new(from: &glam::Vec2, to: &glam::Vec2) -> Self {
                let from = from.round().as_ivec2();
                let to = to.round().as_ivec2();
                let run = to.x - from.x;
                let rise = to.y - from.y;
                let threshold = rise.abs();
                Self {
                    rise,
                    delta: run.abs() * 2,
                    threshold,
                    threshold_inc: threshold * 2,
                    adjust: run.signum(),
                    x: from.x,
                    offset: 0,
                }
            }

            fn step(&mut self) {
                self.offset += self.delta;
                while self.offset >= self.threshold {
                    self.x += self.adjust;
                    self.threshold += self.threshold_inc;
                }
            }
        }

        let mut rasterize_line = |left_interpolator: &Interpolator<A>,
                                  right_interpolator: &Interpolator<A>,
                                  left_data: &BresenhamData,
                                  right_data: &BresenhamData,
                                  y: i32| {
            let (left_zw, left_attributes) = left_interpolator.data();
            let (right_zw, right_attributes) = right_interpolator.data();
            let mut interpolator = Interpolator::from_data(
                left_zw,
                right_zw,
                left_attributes,
                right_attributes,
                right_data.x - left_data.x,
            );
            for x in left_data.x..=right_data.x {
                let (zw, attributes) = interpolator.next().unwrap();
                check_and_perform_action(x, y, zw, attributes);
            }
        };

        let mut left_data = BresenhamData::new(&top.position.xy(), &left.position.xy());
        let mut left_interpolator = Interpolator::new(&top, &left, -left_data.rise);
        let mut right_data = BresenhamData::new(&top.position.xy(), &right.position.xy());
        let mut right_interpolator = Interpolator::new(&top, &right, -right_data.rise);

        let (top_steps, bottom_steps, left_higher) = if left.position.y > right.position.y {
            (-left_data.rise, -right_data.rise - -left_data.rise, true)
        } else {
            (-right_data.rise, -left_data.rise - -right_data.rise, false)
        };

        let mut y = top.position.y as i32;
        for _ in 0..top_steps {
            rasterize_line(
                &left_interpolator,
                &right_interpolator,
                &left_data,
                &right_data,
                y,
            );
            left_interpolator.step();
            left_data.step();
            right_interpolator.step();
            right_data.step();
            y -= 1;
        }

        if left_higher {
            left_data = BresenhamData::new(&left.position.xy(), &right.position.xy());
            left_interpolator = Interpolator::new(&left, &right, -left_data.rise);
        } else {
            right_data = BresenhamData::new(&right.position.xy(), &left.position.xy());
            right_interpolator = Interpolator::new(&right, &left, -right_data.rise);
        }
        if left_data.x > right_data.x {
            mem::swap(&mut left_data, &mut right_data);
            mem::swap(&mut left_interpolator, &mut right_interpolator);
        }

        for _ in 0..bottom_steps {
            rasterize_line(
                &left_interpolator,
                &right_interpolator,
                &left_data,
                &right_data,
                y,
            );
            left_interpolator.step();
            left_data.step();
            right_interpolator.step();
            right_data.step();
            y -= 1;
        }
        rasterize_line(
            &left_interpolator,
            &right_interpolator,
            &left_data,
            &right_data,
            y,
        );
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

    fn data(&self) -> (glam::Vec2, [f32; A]) {
        (self.from_pos, self.from_attrib)
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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CullFace {
    Cw,
    Ccw,
}

#[derive(Copy, Clone, Debug)]
pub struct EdgeFunctionRasterizer {
    width: u32,
    height: u32,
    cull_face: Option<CullFace>,
}

impl EdgeFunctionRasterizer {
    pub fn new(cull_face: Option<CullFace>) -> Self {
        Self {
            width: 0,
            height: 0,
            cull_face,
        }
    }
}

impl<const A: usize, const S: usize> Rasterizer<3, A, S> for EdgeFunctionRasterizer {
    fn set_screen_size(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    fn rasterize<'a, U, T, DF, FS, B>(
        &self,
        shape: [FragmentInput<A>; 3],
        fragment_tools: &mut FragmentTools<'a, U, T, FS, B, A, S>,
        depth_tools: &mut Option<DepthTools<'a, DF, S>>,
        multisample_state: &MultisampleState<S>,
    ) where
        U: Copy,
        T: Copy + Default,
        DF: Fn(f32, f32) -> bool,
        FS: Fn(FragmentInput<A>, U) -> T,
        B: Fn(&T, &T) -> T,
    {
        let v0 = &shape[0];
        let v1 = &shape[1];
        let v2 = &shape[2];

        let area = {
            let a = v0.position.xy() - v1.position.xy();
            let b = v2.position.xy() - v1.position.xy();
            a.x * b.y - a.y * b.x
        };

        let inv_area = if area > 0.0 {
            if self.cull_face == Some(CullFace::Cw) {
                return;
            }
            -1.0 / area
        } else {
            if self.cull_face == Some(CullFace::Ccw) {
                return;
            }
            1.0 / area
        };

        let mut min_bound = v0.position.xy();
        let mut max_bound = v0.position.xy();
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
        let min_bound = min_bound.max(glam::vec2(0.0, 0.0)).as_uvec2();
        let max_bound = max_bound
            .min(glam::vec2(self.width as f32, self.height as f32))
            .ceil()
            .as_uvec2();

        let w0_base = (v1.position.x * v2.position.y - v1.position.y * v2.position.x) * inv_area;
        let w1_base = (v2.position.x * v0.position.y - v2.position.y * v0.position.x) * inv_area;
        let w2_base = (v0.position.x * v1.position.y - v0.position.y * v1.position.x) * inv_area;

        let w0_x_delta = (v1.position.y - v2.position.y) * inv_area;
        let w1_x_delta = (v2.position.y - v0.position.y) * inv_area;
        let w2_x_delta = (v0.position.y - v1.position.y) * inv_area;

        let w0_y_delta = (v2.position.x - v1.position.x) * inv_area;
        let w1_y_delta = (v0.position.x - v2.position.x) * inv_area;
        let w2_y_delta = (v1.position.x - v0.position.x) * inv_area;

        let w0_static_data = StaticData {
            base: w0_base,
            x_delta: w0_x_delta,
            y_delta: w0_y_delta,
        };
        let w1_static_data = StaticData {
            base: w1_base,
            x_delta: w1_x_delta,
            y_delta: w1_y_delta,
        };
        let w2_static_data = StaticData {
            base: w2_base,
            x_delta: w2_x_delta,
            y_delta: w2_y_delta,
        };

        let mut samples = [SampleData {
            w0: BarycentricCoordinate::new(&w0_static_data),
            w1: BarycentricCoordinate::new(&w1_static_data),
            w2: BarycentricCoordinate::new(&w2_static_data),
        }; S];

        for x in min_bound.x..max_bound.x {
            let x_float = x as f32;
            for i in 0..S {
                samples[i].compute_row_bary_coords(x_float + multisample_state.sample_offsets[i].x);
            }
            for y in min_bound.y..max_bound.y {
                let y_float = y as f32;
                let mut covered_samples = [(0, 0.0); S];
                let mut covered_samples_len = 0;
                for i in 0..S {
                    let sample = &mut samples[i];
                    sample.compute_barycentric_coords(
                        y_float + multisample_state.sample_offsets[i].y,
                    );
                    if sample.is_in_triangle() {
                        let depth = sample.interpolate(v0.position.z, v1.position.z, v2.position.z);
                        if 0.0 < depth && depth < 1.0 {
                            let depth_test_passed =
                                depth_tools.as_ref().map_or(true, |depth_tools| {
                                    depth_tools.compare(x, y, i as u32, depth)
                                });
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
                    let (fragment_sample, xy) = match multisample_state.center_offset {
                        PixelCenterOffset::Index(i) => {
                            let xy =
                                glam::vec2(x_float, y_float) + multisample_state.sample_offsets[i];
                            (samples[i], xy)
                        }
                        PixelCenterOffset::Offset(offset) => {
                            let mut fragment_sample = SampleData {
                                w0: BarycentricCoordinate::new(&w0_static_data),
                                w1: BarycentricCoordinate::new(&w1_static_data),
                                w2: BarycentricCoordinate::new(&w2_static_data),
                            };
                            let xy = glam::vec2(x_float, y_float) + offset;
                            fragment_sample.compute_row_bary_coords(xy.x);
                            fragment_sample.compute_barycentric_coords(xy.y);
                            (fragment_sample, xy)
                        }
                    };
                    let zw = fragment_sample.interpolate(
                        v0.position.zw(),
                        v1.position.zw(),
                        v2.position.zw(),
                    );
                    let position = (xy, zw).into();
                    let inv_w = 1.0 / zw[1];
                    let attributes = fragment_sample
                        .interpolate_arr(&v0.attributes, &v1.attributes, &v2.attributes)
                        .map(|attr| attr * inv_w);
                    let fragment_value = fragment_tools.shade_fragment(FragmentInput {
                        position,
                        attributes,
                    });
                    for i in 0..covered_samples_len {
                        let (sample, depth) = unsafe { *covered_samples.get_unchecked(i) };
                        if let Some(depth_tools) = depth_tools {
                            depth_tools.write(x, y, sample, depth);
                        }
                        fragment_tools.blend_and_write(x, y, sample, &fragment_value);
                    }
                }
            }
        }
    }
}

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

    fn compute_row(&mut self, x: f32) {
        self.row_value = self.static_data.x_delta * x + self.static_data.base;
    }

    fn compute_value(&mut self, y: f32) {
        self.value = self.static_data.y_delta * y + self.row_value;
    }
}

#[derive(Copy, Clone)]
struct SampleData<'a> {
    w0: BarycentricCoordinate<'a>,
    w1: BarycentricCoordinate<'a>,
    w2: BarycentricCoordinate<'a>,
}

impl<'a> SampleData<'a> {
    fn compute_row_bary_coords(&mut self, x: f32) {
        self.w0.compute_row(x);
        self.w1.compute_row(x);
        self.w2.compute_row(x);
    }

    fn compute_barycentric_coords(&mut self, y: f32) {
        self.w0.compute_value(y);
        self.w1.compute_value(y);
        self.w2.compute_value(y);
    }

    fn interpolate<T>(&self, v0: T, v1: T, v2: T) -> T
        where
            T: Mul<f32, Output = T> + Add<Output = T>,
    {
        v0 * self.w0.value + v1 * self.w1.value + v2 * self.w2.value
    }

    fn interpolate_arr<T, const N: usize>(
        &self,
        a0: &[T; N],
        a1: &[T; N],
        a2: &[T; N],
    ) -> [T; N]
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
            *res_elem =
                v0_elem * self.w0.value + v1_elem * self.w1.value + v2_elem * self.w2.value;
        }
        res_arr
    }

    fn is_in_triangle(&self) -> bool {
        self.w0.value.to_bits() >> 31 == self.w1.value.to_bits() >> 31
            && self.w1.value.to_bits() >> 31 == self.w2.value.to_bits() >> 31
    }
}
