import taichi as ti
ti.init(ti.cuda)

resolution = (1280,720)
num_roots = 4
iterations = 20 # 50 for better quality
scale = 1
translate_x = 0
translate_y = 0

dtype = ti.f32 # ti.f64 for better quality

roots = ti.Vector.field(2,dtype,shape = num_roots)
roots[0] = ti.Vector([0.5,0])
roots[1] = ti.Vector([-0.5,0])
roots[2] = ti.Vector([0,0.5])
roots[3] = ti.Vector([0,-0.5])


x = ti.Vector.field(2,dtype,shape = resolution)
p = ti.Vector.field(2,dtype,shape = resolution)
p_grad = ti.Vector.field(2,dtype,shape = resolution)


@ti.kernel
def set_x(scale:float,translate_x:float,translate_y:float):
    for i,j in x:
        x0 = float(ti.Vector([i,j])) / resolution[1]
        x0 = (x0 * 2) - 1
        x[i,j] = x0 * scale + ti.Vector([translate_x,translate_y])

@ti.func
def complex_mul(c1,c2):
    return ti.Vector([c1[0]*c2[0] - c1[1]*c2[1], c1[0]*c2[1] + c1[1]*c2[0]])

@ti.func
def complex_div(c1,c2):
    a = c1[0]
    b = c1[1]
    c = c2[0]
    d = c2[1]
    return ti.Vector([a*c+b*d,b*c-a*d]) / (c*c + d*d)

@ti.kernel
def compute_p():
    for i,j in p:
        result = ti.Vector([1.0, 0.0])
        for r in ti.static(range(num_roots)):
            result = complex_mul(result, x[i,j] - roots[r])
        p[i,j] = result

@ti.kernel
def compute_p_grad():
    for i,j in p:
        result = ti.Vector([0.0, 0.0])
        acc = ti.Vector([1.0,0.0])
        for r in range(num_roots):
            f = x[i,j] - roots[r]
            f_grad = ti.Vector([1.0, 0.0])
            g = ti.Vector([1.0, 0.0])
            for r2 in  range(r+1,num_roots):
                g = complex_mul(g, (x[i,j] - roots[r2]))
            result = result + complex_mul(complex_mul(f_grad, g) , acc)
            acc = complex_mul(acc , f)
        p_grad[i,j] = result

@ti.kernel
def newton():
    for i,j in x:
        # x.grad[i,j][1] = 0
        x[i,j] = x[i,j] - complex_div(p[i,j] , p_grad[i,j])

image = ti.Vector.field(3,float,shape = resolution)
roots_colors = ti.Vector.field(3,float,shape = num_roots)
roots_colors[0] = ti.Vector([0.3,0.1,0.7])
roots_colors[1] = ti.Vector([0.7,0.1,0.3])
roots_colors[2] = ti.Vector([0.3,0.7,0.1])
roots_colors[3] = ti.Vector([0.7,0.3,0.1])


@ti.kernel
def render():
    for i,j in image:
        min_dist = -1.0
        min_dist_root = -1
        for r in ti.static(range(num_roots)):
            dist = (x[i,j]-roots[r]).norm()
            if dist < min_dist or min_dist == -1:
                min_dist = dist
                min_dist_root = r
        image[i,j] = roots_colors[min_dist_root]

def update():
    set_x(scale,translate_x,translate_y)
    for i in range(iterations):
        compute_p()
        compute_p_grad()
        newton()
    render()

update()


window = ti.ui.Window("Newton's Fracal",res = resolution)
canvas = window.get_canvas()

while window.running:
    if window.is_pressed('w'):
        scale /= 1.02
    elif window.is_pressed('s'):
        scale *= 1.02
    elif window.is_pressed(ti.GUI.LEFT):
        translate_x -= 0.02 * scale
    elif window.is_pressed(ti.GUI.RIGHT):
        translate_x += 0.02 * scale
    elif window.is_pressed(ti.GUI.UP):
        translate_y += 0.02 * scale
    elif window.is_pressed(ti.GUI.DOWN):
        translate_y -= 0.02 * scale

    with window.GUI.sub_window("Fractal",0.7,0.05,0.25,0.9) as w:
        w.text("Arrows keys to move")
        w.text("W to zoom in")
        w.text("S to zoom out")
        w.text("")
        for r in range(4):
            root_name = f"root {r+1}"
            w.text(root_name)
            curr_root = roots[r]
            curr_root[0] = w.slider_float(root_name+" x",curr_root[0],-1,1)
            curr_root[1] = w.slider_float(root_name+" y",curr_root[1],-1,1)
            curr_color = (roots_colors[r][0],roots_colors[r][1],roots_colors[r][2])
            curr_color = w.color_edit_3(root_name+" color",curr_color)
            roots_colors[r] = ti.Vector([*curr_color])
            w.text("")

    update()

    canvas.set_image(image)
    window.show()
