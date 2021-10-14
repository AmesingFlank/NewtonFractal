import taichi as ti
ti.init(ti.gpu)

resolution = (512,512)
num_roots = 3
iterations = 0

roots = ti.Vector.field(2,float,shape = num_roots)
roots[0] = ti.Vector([0.5,0])
roots[1] = ti.Vector([-0.5,0.5])
roots[2] = ti.Vector([-0.5,-0.5])

x = ti.Vector.field(2,float,shape = resolution,needs_grad=True)
p = ti.Vector.field(2,float,shape = resolution,needs_grad=True)

@ti.kernel
def set_x():
    for i,j in x:
        x0 = float(ti.Vector([i,j])) / resolution
        x0 = (x0 * 2) - 1
        x[i,j] = x0

@ti.func
def complex_mul(c1,c2):
    return ti.Vector([c1[0]*c2[0] - c1[1]*c2[1], c1[0]*c2[1] + c1[1]*c2[0]])

@ti.func
def complex_div(c1,c2):
    a = c1[0]
    b = c1[1]
    c = c2[0]
    d = c2[1]
    # https://mathworld.wolfram.com/ComplexDivision.html
    return ti.Vector([ (a*c+b*d)/(c**2+d**2), (b*c-a*d)/(c**2+d**2)])

@ti.kernel
def compute_p():
    for i,j in p:
        result = ti.Vector([1.0, 1.0])
        for r in ti.static(range(num_roots)):
            result = result * (x[i,j] - roots[r])
        p[i,j] = result

@ti.kernel
def newton():
    for i,j in x:
        x[i,j] = x[i,j] - complex_div(p[i,j] , x.grad[i,j])

image = ti.Vector.field(3,float,shape = resolution)
roots_colors = ti.Vector.field(3,float,shape = num_roots)
roots_colors[0] = ti.Vector([1,0,0])
roots_colors[1] = ti.Vector([0,1,0])
roots_colors[2] = ti.Vector([0,0,1])

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
        # image[i,j][0] = abs(x[i,j][0])
        # image[i,j][1] = abs(x[i,j][1])
        # image[i,j][2] = 0

def update():
    set_x()
    for i in range(iterations):
        compute_p()
        compute_p.grad()
        newton()
    render()

update()

window = ti.GUI("Newton's Fracal",res = resolution)

while window.running:
    window.set_image(image)
    window.show()
