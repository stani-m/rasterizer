#![allow(dead_code)]

use std::collections::HashSet;

use gltf::json::accessor::ComponentType;

fn triangles_to_lines_index(triangles: &[u32]) -> Vec<u32> {
    let mut lines = HashSet::new();
    for triangle in triangles.chunks_exact(3) {
        let mut a = triangle[0];
        let mut b = triangle[1];
        let mut c = triangle[2];
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        if a > c {
            std::mem::swap(&mut a, &mut c);
        }
        if b > c {
            std::mem::swap(&mut b, &mut c);
        }
        lines.insert([a, b]);
        lines.insert([b, c]);
        lines.insert([c, a]);
    }
    lines
        .into_iter()
        .flat_map(|line| line.into_iter())
        .collect()
}

pub struct Model<'a> {
    node: gltf::Node<'a>,
    name: String,
    vertex_buffer: Vec<glam::Vec3>,
    index_buffer: Vec<u32>,
    children: Vec<Model<'a>>,
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
}

impl<'a> Model<'a> {
    pub fn from(gltf: &'a gltf::Document, buffer: &[u8]) -> Self {
        let scene = gltf.default_scene().unwrap();
        let node = scene.nodes().nth(0).unwrap();
        Self::from_node(node, buffer)
    }

    pub fn from_node(node: gltf::Node<'a>, buffer: &[u8]) -> Self {
        let name = node.name().unwrap().to_string();
        if node.mesh().unwrap().primitives().len() != 1 {
            panic!("Multiple primitives not supported!");
        }
        let primitive = node.mesh().unwrap().primitives().nth(0).unwrap();
        let vertex_view = primitive
            .attributes()
            .find_map(|attribute| match attribute.0 {
                gltf::Semantic::Positions => Some(attribute.1),
                _ => None,
            })
            .expect("Position attribute not found!")
            .view()
            .unwrap();
        let vertex_buffer = {
            let offset = vertex_view.offset();
            let length = vertex_view.length();
            let end = offset + length;
            bytemuck::cast_slice(&buffer[offset..end])
                .chunks_exact(3)
                .map(|vertex| glam::Vec3::from_slice(vertex))
                .collect()
        };
        let index_accessor = primitive.indices().unwrap();
        let index_view = index_accessor.view().unwrap();
        let index_buffer = {
            let offset = index_view.offset();
            let length = index_view.length();
            let end = offset + length;
            match index_accessor.data_type() {
                ComponentType::U8 => buffer[offset..end]
                    .iter()
                    .map(|&index| index as u32)
                    .collect(),
                ComponentType::U16 => bytemuck::cast_slice::<u8, u16>(&buffer[offset..end])
                    .into_iter()
                    .map(|&index| index as u32)
                    .collect(),
                ComponentType::U32 => bytemuck::cast_slice(&buffer[offset..end]).to_owned(),
                _ => panic!("Unsupported index accessor data type!"),
            }
        };
        let (translation, rotation, scale) = node.transform().decomposed();
        let translation = translation.into();
        let rotation = glam::Quat::from_array(rotation);
        let scale = scale.into();
        let children = node
            .children()
            .map(|child| Model::from_node(child, buffer))
            .collect();

        Self {
            node,
            name,
            vertex_buffer,
            index_buffer,
            children,
            translation,
            rotation,
            scale,
        }
    }

    pub fn load_vertex_attribute<T: From<[f32; N]>, const N: usize>(
        &self,
        attribute_type: gltf::Semantic,
        buffer: &[u8],
    ) -> Vec<T> {
        let attribute_view = self
            .node
            .mesh()
            .unwrap()
            .primitives()
            .nth(0)
            .unwrap()
            .attributes()
            .find_map(|attribute| (attribute.0 == attribute_type).then(|| attribute.1))
            .expect("Attribute not found!")
            .view()
            .unwrap();
        let start = attribute_view.offset();
        let end = start + attribute_view.length();

        bytemuck::cast_slice(&buffer[start..end])
            .chunks_exact(N)
            .map(|chunk| {
                let array: [f32; N] = chunk.try_into().unwrap();
                array.into()
            })
            .collect()
    }

    /// Calling this multiple times may lead to interesting visual effects
    pub fn transform_triangles_to_lines(&mut self) {
        self.index_buffer = triangles_to_lines_index(&self.index_buffer);
        self.children
            .iter_mut()
            .for_each(Self::transform_triangles_to_lines);
    }

    pub fn vertex_buffer(&self) -> &[glam::Vec3] {
        &self.vertex_buffer
    }

    pub fn index_buffer(&self) -> &[u32] {
        &self.index_buffer
    }

    pub fn model_matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    pub fn children(&self) -> Vec<&Self> {
        let mut children = self.children.iter().collect::<Vec<_>>();
        let children_children = self.children.iter().map(Self::children).flatten();
        children.extend(children_children);
        children
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

fn main() {}
