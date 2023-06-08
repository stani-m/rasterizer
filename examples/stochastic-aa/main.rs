use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use rasterizer::presenter::WgpuPresenter;
use rasterizer::rasterizer::AALineRasterizer;
use rasterizer::{
    blend_function, clipper, Buffer, Color, ListShapeAssembler, Pipeline, PipelineDescriptor,
    StaticMultisampler, StochasticMultisampler, UnshadedFragment,
};

fn main() {
    let width = 800;
    let height = 600;
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Stochastic multisampling")
        .with_inner_size(PhysicalSize::new(width, height))
        .build(&event_loop)
        .unwrap();

    let mut framebuffer = Buffer::new(width, height);
    let mut render_buffer = Buffer::new(width, height);
    let mut presenter = WgpuPresenter::new(&window, width, height, true);
    let mut static_pipeline = Pipeline::new(PipelineDescriptor {
        vertex_shader: |vertex: glam::Vec2, (projection, _)| UnshadedFragment {
            position: projection * glam::vec4(vertex.x, vertex.y, 0.5, 1.0),
            attributes: [],
        },
        shape_assembler: ListShapeAssembler::new(),
        clipper: clipper::simple,
        rasterizer: AALineRasterizer::new(),
        depth_state: None,
        fragment_shader: |_, (_, color)| color,
        blend_function: blend_function::replace,
        multisampler: StaticMultisampler::x4(),
    });
    let mut stochastic_pipeline = Pipeline::new(PipelineDescriptor {
        vertex_shader: |vertex: glam::Vec2, (projection, _)| UnshadedFragment {
            position: projection * glam::vec4(vertex.x, vertex.y, 0.5, 1.0),
            attributes: [],
        },
        shape_assembler: ListShapeAssembler::new(),
        clipper: clipper::simple,
        rasterizer: AALineRasterizer::new(),
        depth_state: None,
        fragment_shader: |_, (_, color)| color,
        blend_function: blend_function::replace,
        multisampler: StochasticMultisampler::<4>::new(),
    });
    let mut use_static = false;

    println!("To switch between static and stochastic multisampling press space");
    println!("Using stochastic multisampling");

    const STEP: usize = 5;
    const INCREMENT: f32 = 8.0;

    let mut lines = generate_lines(width, height, STEP, 0.0);
    lines.append(&mut generate_lines(width, height, STEP, INCREMENT));

    let mut projection =
        glam::Mat4::orthographic_lh(0.0, width as f32, 0.0, height as f32, 0.0, 1.0);

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(size) => {
                let width = size.width;
                let height = size.height;

                framebuffer.resize(width, height);
                render_buffer.resize(width, height);
                presenter.resize(width, height);

                lines.clear();
                lines.append(&mut generate_lines(width, height, STEP, 0.0));
                lines.append(&mut generate_lines(width, height, STEP, INCREMENT));

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
            render_buffer.fill(Color::BLACK);

            if use_static {
                static_pipeline.draw(
                    &lines,
                    (projection, Color::new(1.0, 1.0, 1.0, 1.0)),
                    &mut render_buffer,
                    None,
                );
            } else {
                stochastic_pipeline.draw(
                    &lines,
                    (projection, Color::new(1.0, 1.0, 1.0, 1.0)),
                    &mut render_buffer,
                    None,
                );
            }

            render_buffer.resolve(&mut framebuffer);

            presenter.present(&framebuffer);
        }
        _ => (),
    });
}

fn generate_lines(width: u32, height: u32, step: usize, increment: f32) -> Vec<glam::Vec2> {
    let mut cumulative_increment = 0.0;
    let mut lines = Vec::new();
    for x in (2..width).step_by(step) {
        lines.push(glam::vec2(x as f32 + 0.5, 0.0));
        lines.push(glam::vec2(
            x as f32 + 0.5 + cumulative_increment,
            height as f32,
        ));
        cumulative_increment += increment;
    }
    lines
}
