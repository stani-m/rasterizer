use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use rasterizer::presenter::WgpuPresenter;
use rasterizer::rasterizer::AALineRasterizer;
use rasterizer::{
    blend_function, clipper, Buffer, Color, DepthState, FragmentInput, ListShapeAssembler,
    Pipeline, PipelineDescriptor, StaticMultisampler, StochasticMultisampler,
};

fn main() {
    let width = 800;
    let height = 600;
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Lines example")
        .with_inner_size(PhysicalSize::new(width, height))
        .build(&event_loop)
        .unwrap();

    let mut framebuffer = Buffer::new(width, height);
    let mut render_buffer_1 = Buffer::new(width, height);
    let mut render_buffer_2 = Buffer::new(width, height);
    let mut presenter = WgpuPresenter::new(&window, width, height, true);
    let mut static_pipeline = Pipeline::new(PipelineDescriptor {
        vertex_shader: |vertex: glam::Vec2, (projection, _)| FragmentInput {
            position: projection * glam::vec4(vertex.x, vertex.y, 0.5, 1.0),
            attributes: [],
        },
        shape_assembler: ListShapeAssembler::new(),
        clipper: clipper::simple,
        rasterizer: AALineRasterizer::new(1.0),
        depth_state: None::<DepthState<fn(f32, f32) -> bool>>,
        fragment_shader: |_, (_, color)| color,
        blend_function: blend_function::replace,
        multisampler: StaticMultisampler::x4(),
    });
    let mut stochastic_pipeline = Pipeline::new(PipelineDescriptor {
        vertex_shader: |vertex: glam::Vec2, (projection, _)| FragmentInput {
            position: projection * glam::vec4(vertex.x, vertex.y, 0.5, 1.0),
            attributes: [],
        },
        shape_assembler: ListShapeAssembler::new(),
        clipper: clipper::simple,
        rasterizer: AALineRasterizer::new(1.0),
        depth_state: None::<DepthState<fn(f32, f32) -> bool>>,
        fragment_shader: |_, (_, color)| color,
        blend_function: blend_function::replace,
        multisampler: StochasticMultisampler::<4>::new(),
    });
    let mut use_static = true;

    println!("To switch between static and stochastic multisampling press space");
    println!("Using static multisampling");

    const STEP: usize = 5;
    const INCREMENT: f32 = 8.0;

    let mut lines = straight_lines(width, height, STEP);
    let mut other_lines = the_other_lines(width, height, STEP, INCREMENT);

    let mut projection =
        glam::Mat4::orthographic_lh(0.0, width as f32, 0.0, height as f32, 0.0, 1.0);

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(size) => {
                let width = size.width;
                let height = size.height;

                framebuffer.resize(width, height);
                render_buffer_1.resize(width, height);
                render_buffer_2.resize(width, height);
                presenter.resize(width, height);

                lines = straight_lines(width, height, STEP);
                other_lines = the_other_lines(width, height, STEP, INCREMENT);
                projection =
                    glam::Mat4::orthographic_lh(0.0, width as f32, 0.0, height as f32, 0.0, 1.0);
            }
            WindowEvent::KeyboardInput { input, .. } => {
                if input
                    .virtual_keycode
                    .map_or(false, |key_code| key_code == VirtualKeyCode::Space)
                    && input.state == ElementState::Released
                {
                    use_static = !use_static;
                    if use_static {
                        println!("Using static multisampling");
                    } else {
                        println!("Using stochastic multisampling");
                    }
                }
            }
            _ => (),
        },
        Event::MainEventsCleared => {
            if use_static {
                render_buffer_1.fill(Color::default());

                static_pipeline.draw(
                    &lines,
                    (projection, Color::new(1.0, 1.0, 1.0, 1.0)),
                    &mut render_buffer_1,
                    None,
                );

                static_pipeline.draw(
                    &other_lines,
                    (projection, Color::new(1.0, 1.0, 1.0, 1.0)),
                    &mut render_buffer_1,
                    None,
                );

                render_buffer_1.resolve(&mut framebuffer);
            } else {
                render_buffer_2.fill(Color::default());

                stochastic_pipeline.draw(
                    &lines,
                    (projection, Color::new(1.0, 1.0, 1.0, 1.0)),
                    &mut render_buffer_2,
                    None,
                );

                stochastic_pipeline.draw(
                    &other_lines,
                    (projection, Color::new(1.0, 1.0, 1.0, 1.0)),
                    &mut render_buffer_2,
                    None,
                );

                render_buffer_2.resolve(&mut framebuffer);
            }

            presenter.present(&framebuffer);
        }
        _ => (),
    });
}

fn straight_lines(width: u32, height: u32, step: usize) -> Vec<glam::Vec2> {
    let mut lines = Vec::new();
    for x in (0..width).step_by(step) {
        lines.push(glam::vec2(x as f32 + 0.5, 0.0));
        lines.push(glam::vec2(x as f32 + 0.5, height as f32));
    }
    lines
}

fn the_other_lines(width: u32, height: u32, step: usize, increment: f32) -> Vec<glam::Vec2> {
    let increment_increment = increment;
    let mut increment = 0.0;
    let mut lines = Vec::new();
    for x in (2..width).step_by(step) {
        lines.push(glam::vec2(x as f32 + 0.5, 0.0));
        lines.push(glam::vec2(x as f32 + 0.5 + increment, height as f32));
        increment += increment_increment;
    }
    lines
}
