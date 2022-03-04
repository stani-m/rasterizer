use std::ops::{Index, IndexMut};

pub mod presenter;

#[derive(Clone)]
pub struct Framebuffer {
    color: Vec<Color>,
    width: u32,
    height: u32,
}

impl Framebuffer {
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

    pub fn color_slice(&self) -> &[u8] {
        return bytemuck::cast_slice(self.color.as_slice());
    }

    fn calculate_index(&self, x: u32, y: u32) -> u32 {
        self.width * y + x
    }
}

impl Index<[u32; 2]> for Framebuffer {
    type Output = Color;

    fn index(&self, index: [u32; 2]) -> &Self::Output {
        let index = self.calculate_index(index[0], index[1]) as usize;
        &self.color[index]
    }
}

impl IndexMut<[u32; 2]> for Framebuffer {
    fn index_mut(&mut self, index: [u32; 2]) -> &mut Self::Output {
        let index = self.calculate_index(index[0], index[1]) as usize;
        &mut self.color[index]
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
