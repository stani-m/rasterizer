use std::mem;

use crate::{FragmentInput, Interpolator};

pub trait Rasterizer<const A: usize, const S: usize> {
    fn set_screen_size(&mut self, width: u32, height: u32);
    fn rasterize(&self, shape: [FragmentInput<A>; S]) -> Vec<FragmentInput<A>>;
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

    fn rasterize(&self, shape: [FragmentInput<A>; 2]) -> Vec<FragmentInput<A>> {
        let [mut from, mut to] = shape;
        let [mut x0, mut y0] = from.screen_position.to_array();
        let [mut x1, mut y1] = to.screen_position.to_array();
        let mut run = x1 - x0;
        let mut rise = y1 - y0;
        let width_range = 0..self.width;
        let height_range = 0..self.height;
        let depth_range = 0.0..=1.0;
        let check_and_add_fragment =
            |fragment: FragmentInput<A>, fragments: &mut Vec<FragmentInput<A>>| {
                let [x, y] = fragment.screen_position.to_array();
                let z = fragment.position.z;
                if width_range.contains(&x) & &height_range.contains(&y) & &depth_range.contains(&z)
                {
                    fragments.push(fragment);
                }
            };
        if run == 0 {
            if y0 > y1 {
                mem::swap(&mut y0, &mut y1);
                mem::swap(&mut from, &mut to);
                rise = -rise;
            }
            let mut interpolator = Interpolator::new(&from, &to, rise);
            let mut fragments = Vec::with_capacity(rise as usize);
            for y in y0..=y1 {
                let (position, attributes) = interpolator.next().unwrap();
                check_and_add_fragment(
                    FragmentInput {
                        position,
                        screen_position: glam::ivec2(x0, y),
                        attributes,
                    },
                    &mut fragments,
                );
            }
            fragments
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
                let mut fragments = Vec::with_capacity(run as usize);
                for x in x0..=x1 {
                    let (position, attributes) = interpolator.next().unwrap();
                    check_and_add_fragment(
                        FragmentInput {
                            position,
                            screen_position: glam::ivec2(x, y),
                            attributes,
                        },
                        &mut fragments,
                    );
                    offset += delta;
                    if offset >= threshold {
                        y += adjust;
                        threshold += threshold_inc;
                    }
                }
                fragments
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
                let mut fragments = Vec::with_capacity(rise as usize);
                for y in y0..=y1 {
                    let (position, attributes) = interpolator.next().unwrap();
                    check_and_add_fragment(
                        FragmentInput {
                            position,
                            screen_position: glam::ivec2(x, y),
                            attributes,
                        },
                        &mut fragments,
                    );
                    offset += delta;
                    if offset >= threshold {
                        x += adjust;
                        threshold += threshold_inc;
                    }
                }
                fragments
            }
        }
    }
}
