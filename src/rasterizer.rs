use std::mem;

use glam::Vec4Swizzles;

use crate::FragmentInput;

pub trait Rasterizer<const A: usize, const S: usize> {
    fn set_screen_size(&mut self, width: u32, height: u32);
    fn rasterize(&self, shape: [FragmentInput<A>; S], action: &mut impl FnMut(FragmentInput<A>));
}

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

impl<const A: usize> Rasterizer<A, 2> for BresenhamLineRasterizer {
    fn set_screen_size(&mut self, width: u32, height: u32) {
        self.width = width as i32;
        self.height = height as i32;
    }

    fn rasterize(&self, line: [FragmentInput<A>; 2], action: &mut impl FnMut(FragmentInput<A>)) {
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
                    action(FragmentInput {
                        position: glam::vec4(x as f32, y as f32, zw[0], zw[1]),
                        attributes,
                    });
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

impl<const A: usize> Rasterizer<A, 3> for BresenhamTriangleRasterizer {
    fn set_screen_size(&mut self, width: u32, height: u32) {
        self.width = width as i32;
        self.height = height as i32;
    }

    fn rasterize(
        &self,
        triangle: [FragmentInput<A>; 3],
        action: &mut impl FnMut(FragmentInput<A>),
    ) {
        let width_range = 0..self.width;
        let height_range = 0..self.height;
        let depth_range = 0.0..=1.0;
        let mut check_and_perform_action =
            |x: i32, y: i32, zw: glam::Vec2, attributes: [f32; A]| {
                let z = zw[0];
                if width_range.contains(&x) && height_range.contains(&y) && depth_range.contains(&z)
                {
                    action(FragmentInput {
                        position: glam::vec4(x as f32, y as f32, zw[0], zw[1]),
                        attributes,
                    });
                }
            };

        let mut top = &triangle[0];
        let mut left = &triangle[1];
        let mut right = &triangle[2];

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

pub enum CullFace {
    Cw,
    Ccw,
}

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

    fn edge_function(p: glam::Vec2, (v0, v1): (glam::Vec2, glam::Vec2)) -> f32 {
        let a = p - v0;
        let b = v1 - v0;
        a.x * b.y - a.y * b.x
    }
}

impl<const A: usize> Rasterizer<A, 3> for EdgeFunctionRasterizer {
    fn set_screen_size(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    fn rasterize(&self, shape: [FragmentInput<A>; 3], action: &mut impl FnMut(FragmentInput<A>)) {
        let v0 = &shape[0];
        let v1 = &shape[1];
        let v2 = &shape[2];

        let area = Self::edge_function(v2.position.xy(), (v0.position.xy(), v1.position.xy()));
        match (&self.cull_face, area > 0.0) {
            (Some(CullFace::Cw), true) => return,
            (Some(CullFace::Ccw), false) => return,
            _ => {}
        }

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
        let min_bound = min_bound.floor().as_ivec2();
        let max_bound = max_bound.ceil().as_ivec2();

        for y in min_bound.y..max_bound.y {
            for x in min_bound.x..max_bound.x {
                let p = glam::ivec2(x, y).as_vec2() + 0.5;
                let mut a0 = Self::edge_function(p, (v0.position.xy(), v1.position.xy()));
                let mut a1 = Self::edge_function(p, (v1.position.xy(), v2.position.xy()));
                let mut a2 = Self::edge_function(p, (v2.position.xy(), v0.position.xy()));
                if a0.signum() == a1.signum() && a1.signum() == a2.signum() {
                    a0 /= area;
                    a1 /= area;
                    a2 /= area;
                    let res_zw =
                        v0.position.zw() * a1 + v1.position.zw() * a2 + v2.position.zw() * a0;
                    let inv_w = 1.0 / res_zw[1];
                    let mut res_attribs = [0.0; A];
                    for ((&v0_attrib, &v1_attrib, &v2_attrib), res_attrib) in v0
                        .attributes
                        .iter()
                        .zip(&v1.attributes)
                        .zip(&v2.attributes)
                        .map(|v012_attribs| (v012_attribs.0 .0, v012_attribs.0 .1, v012_attribs.1))
                        .zip(&mut res_attribs)
                    {
                        *res_attrib = (v0_attrib * a1 + v1_attrib * a2 + v2_attrib * a0) * inv_w;
                    }
                    action(FragmentInput {
                        position: (p, res_zw).into(),
                        attributes: res_attribs,
                    })
                }
            }
        }
    }
}
