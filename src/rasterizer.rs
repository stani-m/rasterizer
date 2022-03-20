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

struct Interpolator<const N: usize> {
    from_pos: glam::Vec4,
    pos_delta: glam::Vec4,
    from_attrib: [f32; N],
    attrib_delta: [f32; N],
}

impl<const A: usize> Interpolator<A> {
    fn new(from: &FragmentInput<A>, to: &FragmentInput<A>, steps: i32) -> Self {
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

impl<const A: usize> Iterator for Interpolator<A> {
    type Item = (glam::Vec4, [f32; A]);

    fn next(&mut self) -> Option<Self::Item> {
        let w = 1.0 / self.from_pos.w;
        let position = (self.from_pos.xy() * w, self.from_pos.z, self.from_pos.w).into();
        let attributes = self.from_attrib.map(|attrib| attrib * w);
        let res = Some((position, attributes));
        self.from_pos += self.pos_delta;
        for (attr, delta) in self.from_attrib.iter_mut().zip(&self.attrib_delta) {
            *attr += delta;
        }
        res
    }
}
