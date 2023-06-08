use std::time::Instant;

use winit::dpi::PhysicalSize;
use winit::event::{Event, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::WindowBuilder;

use model::Model;
use rasterizer::presenter::WgpuPresenter;
use rasterizer::rasterizer::EdgeFunctionTiledRasterizer;
use rasterizer::{
    blend_function, clipper, Buffer, Color, DepthState, ListShapeAssembler, Pipeline,
    PipelineDescriptor, StaticMultisampler, UnshadedFragment,
};

#[path = "../model.rs"]
mod model;

fn main() {
    let width = 800;
    let height = 600;
    let mut event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Clipping example")
        .with_inner_size(PhysicalSize::new(width, height))
        .build(&event_loop)
        .unwrap();

    let mut framebuffer = Buffer::new(width, height);
    let mut render_buffer = Buffer::new(width, height);
    let mut depth_buffer = Buffer::new(width, height);
    let mut presenter = WgpuPresenter::new(&window, width, height, false);
    let mut pipeline = Pipeline::new(PipelineDescriptor {
        vertex_shader:
            |(position, uv, normal): (glam::Vec3, glam::Vec2, glam::Vec3),
             (camera, model, normal_matrix): (glam::Mat4, glam::Mat4, glam::Mat3)| {
                let normal = normal_matrix * normal;
                let position = model * glam::Vec4::from((position, 1.0));

                UnshadedFragment {
                    position: camera * position,
                    attributes: [
                        uv.x, uv.y, normal.x, normal.y, normal.z, position.x, position.y,
                        position.z,
                    ],
                }
            },
        shape_assembler: ListShapeAssembler::new(),
        clipper: clipper::sutherland_hodgman_near_far,
        rasterizer: EdgeFunctionTiledRasterizer::<16>::new(None, 0),
        depth_state: Some(DepthState {
            depth_function: f32::le,
            write_depth: true,
        }),
        fragment_shader: |fragment: UnshadedFragment<8>, _| -> Color {
            const LIGHT_POSITION: glam::Vec3 = glam::vec3(3.0, 4.0, 12.0);
            const LIGHT_COLOR: glam::Vec3 = glam::Vec3::ONE;
            const EYE: glam::Vec3 = glam::vec3(0.0, 1.0, 3.0);

            let uv = glam::Vec2::from_slice(&fragment.attributes);
            let normal = glam::Vec3::from_slice(&fragment.attributes[2..]).normalize();
            let position = glam::Vec3::from_slice(&fragment.attributes[5..]);

            let pos = (uv * 24.0).as_ivec2() % 2;
            let object_color = if pos.x == pos.y {
                glam::vec3(0.0, 1.0, 1.0)
            } else {
                glam::vec3(1.0, 0.0, 1.0)
            };

            let light_direction = (LIGHT_POSITION - position).normalize();

            let ambient = LIGHT_COLOR * 0.15;

            let diffuse = LIGHT_COLOR * light_direction.dot(normal).max(0.0);

            let view_dir = (EYE - position).normalize();
            let reflect_dir = -light_direction - 2.0 * normal.dot(-light_direction) * normal;
            let spec = view_dir.dot(reflect_dir).max(0.0).powi(32);
            let specular = spec * LIGHT_COLOR * 0.7;

            glam::Vec4::from((
                (ambient + diffuse) * object_color + specular * LIGHT_COLOR,
                1.0,
            ))
            .into()
        },
        blend_function: blend_function::replace,
        multisampler: StaticMultisampler::x4(),
    });

    let (document, buffers, _) = gltf::import("examples/assets/Cube.gltf")
        .expect("glTF import failed, file probably not found, make sure to run examples from crate root directory");
    let mut cube = Model::new(&document, &buffers[0]);
    let uvs: Vec<glam::Vec2> =
        cube.load_vertex_attribute(gltf::Semantic::TexCoords(0), &buffers[0]);
    let mut vertex_buffer = Vec::with_capacity(cube.index_buffer().len());
    for indices in cube.index_buffer().chunks(3) {
        let i0 = indices[0] as usize;
        let i1 = indices[1] as usize;
        let i2 = indices[2] as usize;

        let v0 = cube.vertex_buffer()[i0];
        let v1 = cube.vertex_buffer()[i1];
        let v2 = cube.vertex_buffer()[i2];

        let uv0 = uvs[i0];
        let uv1 = uvs[i1];
        let uv2 = uvs[i2];

        let normal = (v1 - v0).cross(v2 - v0).normalize();

        vertex_buffer.push((v0, uv0, normal));
        vertex_buffer.push((v1, uv1, normal));
        vertex_buffer.push((v2, uv2, normal));
    }
    let rotate = glam::Quat::from_axis_angle(glam::vec3(0.0, 1.0, 0.0), 1.0);
    cube.rotation *= rotate;
    cube.translation += glam::vec3(0.0, 0.2, 0.0);

    let mut z_offset = 0.9;
    let zoom_amount = 1.0;
    let zoom = glam::Mat4::from_scale(glam::vec3(zoom_amount, zoom_amount, zoom_amount));
    let view = glam::Mat4::look_at_lh(
        glam::vec3(0.0, 1.0, 3.0),
        glam::vec3(0.0, 0.0, 0.0),
        glam::vec3(0.0, 1.0, 0.0),
    );
    let projection =
        glam::Mat4::perspective_lh(45_f32.to_radians(), width as f32 / height as f32, 3.1, 3.9);
    let z_move = glam::Mat4::from_translation(glam::vec3(0.0, 0.0, z_offset));
    let mut camera = projection * z_move * view * zoom;

    let program_start = Instant::now();
    let mut last_frame_time = program_start;
    let mut last_second = 0;
    let mut frames = 0u32;

    event_loop.run_return(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(size) => {
                let width = size.width;
                let height = size.height;
                framebuffer.resize(width, height);
                render_buffer.resize(width, height);
                depth_buffer.resize(width, height);
                presenter.resize(width, height);
                let projection = glam::Mat4::perspective_lh(
                    45_f32.to_radians(),
                    width as f32 / height as f32,
                    3.1,
                    3.9,
                );
                camera = projection * z_move * view * zoom;
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                MouseScrollDelta::LineDelta(_, y) => {
                    z_offset += y * 0.1;
                    let z_move = glam::Mat4::from_translation(glam::vec3(0.0, 0.0, z_offset));
                    camera = projection * z_move * view * zoom;
                }
                _ => {}
            },
            _ => (),
        },
        Event::MainEventsCleared => {
            let current_frame_time = Instant::now();
            let delta_time = current_frame_time - last_frame_time;
            let since_program_start = current_frame_time - program_start;

            let current_second = since_program_start.as_secs();
            if current_second != last_second {
                println!("FPS: {frames}");
                last_second = current_second;
                frames = 0;
            }
            frames += 1;

            let rotate = glam::Quat::from_axis_angle(
                glam::vec3(0.0, 1.0, 0.0),
                -delta_time.as_secs_f32() * 0.1,
            );
            cube.rotation *= rotate;
            let model = cube.model_matrix();

            let normal_matrix = glam::Mat3::from_mat4(model).inverse().transpose();

            render_buffer.fill(Color::BLACK);
            depth_buffer.fill(1.0);

            pipeline.draw(
                &vertex_buffer,
                (camera, model, normal_matrix),
                &mut render_buffer,
                Some(&mut depth_buffer),
            );

            render_buffer.resolve(&mut framebuffer);

            presenter.present(&framebuffer);

            last_frame_time = current_frame_time;
        }
        _ => (),
    });
}
