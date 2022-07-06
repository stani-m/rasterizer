use std::time::Instant;

use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::WindowBuilder;

use model::Model;
use rasterizer::presenter::WgpuPresenter;
use rasterizer::rasterizer::BresenhamLineRasterizer;
use rasterizer::{
    blend_function, clipper, depth_function, Buffer, Color, DepthState, FragmentInput,
    ListShapeAssembler, Pipeline, PipelineDescriptor,
};

#[path = "../model.rs"]
mod model;

fn main() {
    let width = 800;
    let height = 600;
    let mut event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Lines example")
        .with_inner_size(PhysicalSize::new(width, height))
        .build(&event_loop)
        .unwrap();

    let mut render_buffer = Buffer::<Color, 1>::new(width, height);
    let mut depth_buffer = Buffer::<f32, 1>::new(width, height);
    let mut presenter = WgpuPresenter::new(&window, width, height, true);
    let mut pipeline = Pipeline::new(PipelineDescriptor {
        vertex_shader: |vertex, (transform, _)| {
            let position: glam::Vec4 = (vertex, 1.0).into();
            FragmentInput {
                position: transform * position,
                attributes: [],
            }
        },
        shape_assembler: ListShapeAssembler::new(),
        clipper: clipper::simple,
        rasterizer: BresenhamLineRasterizer::new(),
        depth_state: DepthState {
            depth_function: depth_function::less_or_equal,
            write_depth: true,
        },
        fragment_shader: |_, (_, color)| color,
        blend_function: blend_function::replace,
    });

    let (document, buffers, _) = gltf::import("examples/assets/TheDonut.gltf")
        .expect("glTF import failed, file probably not found, make sure to run examples from crate root directory");
    let mut donut = Model::from(&document, &buffers[0]);
    donut.transform_triangles_to_lines();

    let zoom = glam::Mat4::from_scale(glam::vec3(20.0, 20.0, 20.0));
    let view = glam::Mat4::look_at_lh(
        glam::vec3(1.0, 2.0, 3.0),
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

            render_buffer.fill(Color::default());
            depth_buffer.fill(1.0);
            let transform = camera * donut.model_matrix();
            pipeline.draw_indexed(
                donut.vertex_buffer(),
                donut.index_buffer(),
                (transform, Color::new(0.0, 1.0, 1.0, 1.0)),
                &mut render_buffer,
                &mut depth_buffer,
            );
            for child in donut.children() {
                pipeline.draw_indexed(
                    child.vertex_buffer(),
                    child.index_buffer(),
                    (
                        transform * child.model_matrix(),
                        Color::new(1.0, 0.0, 1.0, 1.0),
                    ),
                    &mut render_buffer,
                    &mut depth_buffer,
                );
            }

            presenter.present(&render_buffer);

            last_frame_time = current_frame_time;
        }
        _ => (),
    });
}
