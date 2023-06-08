use std::time::Instant;

use gltf::Semantic;
use winit::dpi::PhysicalSize;
use winit::event::VirtualKeyCode::Space;
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::WindowBuilder;

use model::Model;
use rasterizer::presenter::WgpuPresenter;
use rasterizer::rasterizer::{EdgeFunctionMTRasterizer, Face};
use rasterizer::{
    blend_function, clipper, Buffer, Color, DepthState, ListShapeAssembler, Pipeline,
    PipelineDescriptor, StaticMultisampler, UnshadedFragment,
};

#[path = "../model.rs"]
mod model;

fn main() {
    let width = 1000;
    let height = 700;
    let mut event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Phong example")
        .with_inner_size(PhysicalSize::new(width, height))
        .build(&event_loop)
        .unwrap();

    let mut framebuffer = Buffer::new(width, height);
    let mut render_buffer = Buffer::new(width, height);
    let mut depth_buffer = Buffer::new(width, height);
    let mut presenter = WgpuPresenter::new(&window, width, height, true);
    let mut pipeline = Pipeline::new(PipelineDescriptor {
        vertex_shader: |(position, normal): (glam::Vec3, glam::Vec3),
                        (camera, model, normal_matrix, _): (
            glam::Mat4,
            glam::Mat4,
            glam::Mat3,
            glam::Vec3,
        )| {
            let normal = normal_matrix * normal;
            let position = model * glam::Vec4::from((position, 1.0));

            UnshadedFragment {
                position: camera * position,
                attributes: [
                    normal.x, normal.y, normal.z, position.x, position.y, position.z,
                ],
            }
        },
        shape_assembler: ListShapeAssembler::new(),
        clipper: clipper::sutherland_hodgman_near_far,
        rasterizer: EdgeFunctionMTRasterizer::new(Some(Face::Ccw), 0),
        depth_state: Some(DepthState {
            depth_function: f32::le,
            write_depth: true,
        }),
        fragment_shader: |fragment: UnshadedFragment<6>,
                          (_, _, _, color): (glam::Mat4, glam::Mat4, glam::Mat3, glam::Vec3)|
         -> Color {
            const LIGHT_POSITION: glam::Vec3 = glam::vec3(3.0, 4.0, 12.0);
            const LIGHT_COLOR: glam::Vec3 = glam::Vec3::ONE;
            const EYE: glam::Vec3 = glam::vec3(0.0, 1.0, 3.0);

            let normal = glam::Vec3::from_slice(&fragment.attributes).normalize();
            let position = glam::Vec3::from_slice(&fragment.attributes[3..]);

            let light_direction = (LIGHT_POSITION - position).normalize();

            let ambient = LIGHT_COLOR * 0.15;

            let diffuse = LIGHT_COLOR * light_direction.dot(normal).max(0.0);

            let view_dir = (EYE - position).normalize();
            let reflect_dir = -light_direction - 2.0 * normal.dot(-light_direction) * normal;
            let spec = view_dir.dot(reflect_dir).max(0.0).powi(32);
            let specular = spec * LIGHT_COLOR * 0.7;

            glam::Vec4::from(((ambient + diffuse) * color + specular, 1.0)).into()
        },
        blend_function: blend_function::replace,
        multisampler: StaticMultisampler::x4(),
    });

    let (document, buffers, _) = gltf::import("examples/assets/TheDonut.gltf")
        .expect("glTF import failed, file probably not found, make sure to run examples from crate root directory");
    let buffer = &buffers[0];

    let mut donut = Model::new(&document, buffer);
    donut.translation += glam::vec3(0.0, -0.01, 0.0);

    let donut_normals: Vec<glam::Vec3> =
        donut.load_vertex_attribute(Semantic::Normals, &buffers[0]);
    let donut_buffer = donut
        .vertex_buffer()
        .into_iter()
        .map(ToOwned::to_owned)
        .zip(donut_normals)
        .collect::<Vec<_>>();

    let icing_normals: Vec<glam::Vec3> =
        donut.children()[0].load_vertex_attribute(Semantic::Normals, &buffers[0]);
    let icing_buffer = donut.children()[0]
        .vertex_buffer()
        .into_iter()
        .map(ToOwned::to_owned)
        .zip(icing_normals)
        .collect::<Vec<_>>();

    let zoom_amount = 18.0;
    let zoom = glam::Mat4::from_scale(glam::vec3(zoom_amount, zoom_amount, zoom_amount));
    let view = glam::Mat4::look_at_lh(
        glam::vec3(0.0, 1.0, 3.0),
        glam::vec3(0.0, 0.0, 0.0),
        glam::vec3(0.0, 1.0, 0.0),
    );
    let projection = glam::Mat4::perspective_lh(
        45_f32.to_radians(),
        width as f32 / height as f32,
        0.1,
        100.0,
    );
    let mut camera = projection * view * zoom;

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
                    0.1,
                    100.0,
                );
                camera = projection * view * zoom;
            }
            WindowEvent::KeyboardInput { input, .. }
                if input.virtual_keycode == Some(Space)
                    && input.state == ElementState::Released => {}
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
            donut.rotation *= rotate;

            let donut_model = donut.model_matrix();
            let donut_normal_matrix = glam::Mat3::from_mat4(donut_model).inverse().transpose();

            let icing_model = donut_model * donut.children()[0].model_matrix();
            let icing_normal_matrix = glam::Mat3::from_mat4(icing_model).inverse().transpose();

            render_buffer.fill(Color::BLACK);
            depth_buffer.fill(1.0);

            pipeline.draw_indexed(
                &donut_buffer,
                donut.index_buffer(),
                (
                    camera,
                    donut_model,
                    donut_normal_matrix,
                    glam::vec3(0.0, 1.0, 1.0),
                ),
                &mut render_buffer,
                Some(&mut depth_buffer),
            );

            pipeline.draw_indexed(
                &icing_buffer,
                donut.children()[0].index_buffer(),
                (
                    camera,
                    icing_model,
                    icing_normal_matrix,
                    glam::vec3(1.0, 0.0, 1.0),
                ),
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
