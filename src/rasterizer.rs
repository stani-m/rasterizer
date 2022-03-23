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
        let [mut x0, mut y0] = from.screen_position.to_array();
        let [mut x1, mut y1] = to.screen_position.to_array();
        let mut run = x1 - x0;
        let mut rise = y1 - y0;
        let width_range = 0..self.width;
        let height_range = 0..self.height;
        let depth_range = 0.0..=1.0;
        let mut check_and_perform_action = |fragment: FragmentInput<A>| {
            let [x, y] = fragment.screen_position.to_array();
            let z = fragment.position.z;
            if width_range.contains(&x) && height_range.contains(&y) && depth_range.contains(&z) {
                action(fragment);
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
                let (position, attributes) = interpolator.next().unwrap();
                check_and_perform_action(FragmentInput {
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
                    run = -run;
                    y = y1;
                } else {
                    y = y0;
                }
                let mut interpolator = Interpolator::new(&from, &to, run);
                for x in x0..=x1 {
                    let (position, attributes) = interpolator.next().unwrap();
                    check_and_perform_action(FragmentInput {
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
                    rise = -rise;
                    x = x1;
                } else {
                    x = x0;
                }
                let mut interpolator = Interpolator::new(&from, &to, rise);
                for y in y0..=y1 {
                    let (position, attributes) = interpolator.next().unwrap();
                    check_and_perform_action(FragmentInput {
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
        let mut check_and_perform_action = |fragment: FragmentInput<A>| {
            let [x, y] = fragment.screen_position.to_array();
            let z = fragment.position.z;
            if width_range.contains(&x) && height_range.contains(&y) && depth_range.contains(&z) {
                action(fragment);
            }
        };

        let mut top = &triangle[0];
        let mut left = &triangle[1];
        let mut right = &triangle[2];

        if left.screen_position.y > top.screen_position.y {
            mem::swap(&mut top, &mut left);
        }
        if right.screen_position.y > top.screen_position.y {
            mem::swap(&mut top, &mut right);
        }
        if left.screen_position.x < top.screen_position.x {
            if right.screen_position.x < top.screen_position.x {
                let [left_run, left_rise] = (left.position.xy() - top.position.xy()).to_array();
                let left_slope = left_run / left_rise;
                let [right_run, right_rise] = (right.position.xy() - top.position.xy()).to_array();
                let right_slope = right_run / right_rise;
                if left_slope < right_slope {
                    mem::swap(&mut left, &mut right);
                }
            }
        }
        if left.screen_position.x >= top.screen_position.x {
            if right.screen_position.x > top.screen_position.x {
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
            fn new(from: &glam::IVec2, to: &glam::IVec2) -> Self {
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
            let (left_position, left_attributes) = left_interpolator.data();
            let (right_position, right_attributes) = right_interpolator.data();
            let mut interpolator = Interpolator::from_data(
                left_position,
                right_position,
                left_attributes,
                right_attributes,
                right_data.x - left_data.x,
            );
            for x in left_data.x..=right_data.x {
                let (position, attributes) = interpolator.next().unwrap();
                check_and_perform_action(FragmentInput {
                    position,
                    screen_position: glam::ivec2(x, y),
                    attributes,
                });
            }
        };

        let mut left_data = BresenhamData::new(&top.screen_position, &left.screen_position);
        let mut left_interpolator = Interpolator::new(&top, &left, -left_data.rise);
        let mut right_data = BresenhamData::new(&top.screen_position, &right.screen_position);
        let mut right_interpolator = Interpolator::new(&top, &right, -right_data.rise);

        let (top_steps, bottom_steps, left_higher) =
            if left.screen_position.y > right.screen_position.y {
                (-left_data.rise, -right_data.rise - -left_data.rise, true)
            } else {
                (-right_data.rise, -left_data.rise - -right_data.rise, false)
            };

        let mut y = top.screen_position.y;
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
            left_data = BresenhamData::new(&left.screen_position, &right.screen_position);
            left_interpolator = Interpolator::new(&left, &right, -left_data.rise);
        } else {
            right_data = BresenhamData::new(&right.screen_position, &left.screen_position);
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
    from_pos: glam::DVec4,
    pos_delta: glam::DVec4,
    from_attrib: [f64; A],
    attrib_delta: [f64; A],
}

impl<const A: usize> Interpolator<A> {
    fn new(from: &FragmentInput<A>, to: &FragmentInput<A>, steps: i32) -> Self {
        Self::from_data(
            from.position.as_dvec4(),
            to.position.as_dvec4(),
            from.attributes.map(|attrib| attrib as f64),
            to.attributes.map(|attrib| attrib as f64),
            steps,
        )
    }

    fn from_data(
        from_pos: glam::DVec4,
        to_pos: glam::DVec4,
        from_attrib: [f64; A],
        to_attrib: [f64; A],
        steps: i32,
    ) -> Self {
        let steps = steps as f64;

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

    fn data(&self) -> (glam::DVec4, [f64; A]) {
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
    type Item = (glam::Vec4, [f32; A]);

    fn next(&mut self) -> Option<Self::Item> {
        let w = 1.0 / self.from_pos.w;
        let position: glam::DVec4 =
            (self.from_pos.xy() * w, self.from_pos.z, self.from_pos.w).into();
        let attributes = self.from_attrib.map(|attrib| (attrib * w) as f32);
        let res = Some((position.as_vec4(), attributes));
        self.step();
        res
    }
}
