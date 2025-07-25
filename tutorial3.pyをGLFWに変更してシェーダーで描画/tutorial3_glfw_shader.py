#ODE-0.16.4のtutorial3.pyを書き換えたコードです。
#https://hiramatsuyuusuke.github.io/portfolio2/product3.html

import sys, os, random, time
from math import *
from OpenGL.GL import *
import glfw
import glm
import numpy as np

import ode


# Vertex Shader
vertex_shader_source = """
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;

void main()
{
    FragPos = vec3(model * vec4(position, 1.0));
    Normal = mat3(transpose(inverse(model))) * normal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

# Fragment Shader
fragment_shader_source = """
#version 330 core

in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform vec3 lightDir;    // 光源の方向（正規化済み）
uniform vec3 lightColor;
uniform vec3 objectColor;
uniform vec3 viewPos;

void main()
{
    // 法線ベクトルの正規化
    vec3 norm = normalize(Normal);

    // 光源方向は逆ベクトルで受け取ることが多いのでadjustしてください
    vec3 lightDirection = normalize(-lightDir);

    // ディフューズ光：光の方向と法線の内積
    float diff = max(dot(norm, lightDirection), 0.0);

    // 環境光のみ簡易実装
    vec3 ambient = 0.5 * lightColor;

    // 拡散光強度分だけ色乗算
    vec3 diffuse = diff * lightColor;

    vec3 result = (ambient + diffuse) * objectColor;
    FragColor = vec4(result, 1.0);
}
"""

#シェーダコンパイル
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    # Check compilation
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile failed: {error}")
    return shader

#シェーダプログラム作成
def create_shader_program():
    vertex_shader = compile_shader(vertex_shader_source, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_shader_source, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    # Check linking
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program linking failed: {error}")
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program

# シェーダー用の頂点データと法線データを作成
def draw_body(body, body_index, vertices, indices):
    """Draw an ODE body.
    """
    #boxの頂点座標の計算
    lx,ly,lz = body.boxsize
    box_vx = lx * 0.5
    box_vy = ly * 0.5
    box_vz = lz * 0.5

    #回転前の頂点座標データ
    v = []
    # back face
    v.append( glm.vec3(-box_vx, -box_vy, -box_vz) )   #0
    v.append( glm.vec3( box_vx, -box_vy, -box_vz) )   #1
    v.append( glm.vec3( box_vx,  box_vy, -box_vz) )   #2
    v.append( glm.vec3(-box_vx,  box_vy, -box_vz) )   #3
    # front face
    v.append( glm.vec3(-box_vx, -box_vy,  box_vz) )   #4
    v.append( glm.vec3( box_vx, -box_vy,  box_vz) )   #5
    v.append( glm.vec3( box_vx,  box_vy,  box_vz) )   #6
    v.append( glm.vec3(-box_vx,  box_vy,  box_vz) )   #7

   #回転前の頂点の法線データ（頂点の法線）
    n = []
    # back face
    n.append(glm.vec3( -1, -1, -1))    # 0
    n.append(glm.vec3(  1, -1, -1))    # 1
    n.append(glm.vec3(  1,  1, -1))    # 2
    n.append(glm.vec3( -1,  1, -1))    # 3
    # front face
    n.append(glm.vec3( -1, -1,  1))    # 4
    n.append(glm.vec3(  1, -1,  1))    # 5
    n.append(glm.vec3(  1,  1,  1))    # 6
    n.append(glm.vec3( -1,  1,  1))    # 7

    # glm用（後でデータを入れ替える）の回転行列を作成
    rotation_matrix = glm.rotate(glm.mat4(1.0), glm.radians(0), glm.vec3(0, 0, 1))

    #姿勢データを取得
    R = body.getRotation()
    #rot = np.array([[R[0], R[1], R[2], 0.0],
    #                [R[3], R[4], R[5], 0.0],
    #                [R[6], R[7], R[8], 0.0],
    #                [   0,    0,    0, 1.0]])
    
    # glm用の回転行列に姿勢データを入れる
    rotation_matrix[0,0] = R[0]
    rotation_matrix[1,0] = R[1]
    rotation_matrix[2,0] = R[2]

    rotation_matrix[0,1] = R[3]
    rotation_matrix[1,1] = R[4]
    rotation_matrix[2,1] = R[5]

    rotation_matrix[0,2] = R[6]
    rotation_matrix[1,2] = R[7]
    rotation_matrix[2,2] = R[8]

    # 回転行列からクォータニオンを生成
    quaternion = glm.quat_cast(rotation_matrix)

    #頂点座標の回転変換
    vpx = []
    vpy = []
    vpz = []
    for i in range(8):
        rotated_vector = quaternion * v[i]    # 頂点座標を回転
        x,y,z = rotated_vector #回転後の頂点座標
        vpx.append(x)
        vpy.append(y)
        vpz.append(z)

    #頂点の法線の回転変換
    nx = []
    ny = []
    nz = []    
    for i in range(8):
        rotated_vector = quaternion * n[i]    # 頂点の法線を回転
        x,y,z = glm.normalize(rotated_vector)
        nx.append(x)
        ny.append(y)
        nz.append(z)
        
    # Cube vertices and normals (position XYZ + normals)
    px,py,pz = body.getPosition()   #boxの座標を取得
    arr1 = np.array([], dtype=np.float32)
    for i in range(8):
                         # positions                        # normals
        arr2 = np.array([ vpx[i]+px, vpy[i]+py, vpz[i]+pz,  nx[i], ny[i], nz[i]], dtype=np.float32)
        arr1 = np.append(arr1, arr2)
    vertices_result = np.append(vertices, arr1)
    
    # Indices defining the 12 triangles composing the cube
    i = body_index*8
    arr3 = np.array([
        0+i,1+i,2+i, 2+i,3+i,0+i,  # back face
        4+i,5+i,6+i, 6+i,7+i,4+i,  # front face
        4+i,5+i,1+i, 1+i,0+i,4+i,  # bottom face
        6+i,7+i,3+i, 3+i,2+i,6+i,  # top face
        4+i,7+i,3+i, 3+i,0+i,4+i,  # left face
        5+i,6+i,2+i, 2+i,1+i,5+i   # right face
    ], dtype=np.uint32)
    indices_result = np.append(indices, arr3)

    return vertices_result, indices_result

# geometric utility functions
def scalp (vec, scal):
    vec[0] *= scal
    vec[1] *= scal
    vec[2] *= scal

def length (vec):
    return sqrt (vec[0]**2 + vec[1]**2 + vec[2]**2)

# create_box
def create_box(world, space, density, lx, ly, lz):
    """Create a box body and its corresponding geom."""

    # Create body
    body = ode.Body(world)
    M = ode.Mass()
    M.setBox(density, lx, ly, lz)
    body.setMass(M)

    # Set parameters for drawing the body
    body.shape = "box"
    body.boxsize = (lx, ly, lz)

    # Create a box geom for collision detection
    geom = ode.GeomBox(space, lengths=body.boxsize)
    geom.setBody(body)

    return body, geom

# drop_object
def drop_object():
    """Drop an object into the scene."""

    global bodies, geoms, counter, objcount

    body, geom = create_box(world, space, 1000, 1.0,0.2,0.2)
    body.setPosition( (random.gauss(0,0.03),3.0,random.gauss(0,0.03)) )
    theta = random.uniform(0,2*pi)
    ct = cos (theta)
    st = sin (theta)
    body.setRotation([ct, 0., -st, 0., 1., 0., st, 0., ct])
    bodies.append(body)
    geoms.append(geom)
    counter=0
    objcount+=1

# explosion
def explosion():
    """Simulate an explosion.

    Every object is pushed away from the origin.
    The force is dependent on the objects distance from the origin.
    """
    global bodies

    for b in bodies:
        l=b.getPosition ()
        d = length (l)
        a = max(0, 40000*(1.0-0.2*d*d))
        l = [l[0] / 4, l[1], l[2] /4]
        scalp (l, a / length (l))
        b.addForce(l)

# pull
def pull():
    """Pull the objects back to the origin.

    Every object will be pulled back to the origin.
    Every couple of frames there'll be a thrust upwards so that
    the objects won't stick to the ground all the time.
    """
    global bodies, counter

    for b in bodies:
        l=list (b.getPosition ())
        scalp (l, -1000 / length (l))
        b.addForce(l)
        if counter%60==0:
            b.addForce((0,10000,0))

# Collision callback
def near_callback(args, geom1, geom2):
    """Callback function for the collide() method.

    This function checks if the given geoms do collide and
    creates contact joints if they do.
    """

    # Check if the objects do collide
    contacts = ode.collide(geom1, geom2)

    # Create contact joints
    world,contactgroup = args
    for c in contacts:
        c.setBounce(0.2)
        c.setMu(5000)
        j = ode.ContactJoint(world, contactgroup, c)
        j.attach(geom1.getBody(), geom2.getBody())


######################################################################

# Create a world object
world = ode.World()
world.setGravity( (0,-9.81,0) )
world.setERP(0.8)
world.setCFM(1E-5)

# Create a space object
space = ode.Space()

# Create a plane geom which prevent the objects from falling forever
floor = ode.GeomPlane(space, (0,1,0), 0)   

# A list with ODE bodies
bodies = []

# The geoms for each of the bodies
geoms = []

# A joint group for the contact joints that are generated whenever
# two bodies collide
contactgroup = ode.JointGroup()

# Some variables used inside the simulation loop
fps = 50
dt = 1.0/fps
running = True
state = 0
counter = 0
objcount = 0
lasttime = time.time()


#シェーダーで描画
def use_shader_in_tutorial3(window, shader_program, VAO, VBO, EBO):

    vertices = np.array([], dtype=np.float32)
    indices = np.array([], dtype=np.uint32)

    for index, b in enumerate(bodies):
        #bodyの頂点データを作成
        vertices, indices = draw_body(b, index, vertices, indices)

    glBindVertexArray(VAO)

    # Vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Element buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    # Projection matrix (perspective)
    projection = np.identity(4, dtype=np.float32)

    fov = 45
    aspect_ratio = 800 / 600
    near = 1.0
    far = 100.0

    f = 1.0 / tan(radians(fov) / 2)
    projection[0, 0] = f / aspect_ratio
    projection[1, 1] = f
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = (2 * far * near) / (near - far)
    projection[3, 2] = -1
    projection[3, 3] = 0
        
    # View matrix (camera)
    view = np.identity(4, dtype=np.float32)
    view[3, 0] = 0.0  # Move on x axis
    view[3, 1] = -1.0  # Move on y axis
    view[3, 2] = -4.0  # Move on z axis

    # Start render 
    #while not glfw.window_should_close(window):
    glfw.poll_events()

    glClearColor(0.2, 0.3, 0.3, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(shader_program)

    # Calculate rotation angle
    time = glfw.get_time()
    angle = 0

    # Model matrix: rotate cube over time
    model = np.identity(4, dtype=np.float32)
    c = cos(angle)
    s = sin(angle)

    # Rotation around Y axis
    model[0, 0] = c
    model[0, 2] = s
    model[2, 0] = -s
    model[2, 2] = c

    # Set uniform matrices
    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    proj_loc = glGetUniformLocation(shader_program, "projection")
    light_dir_loc = glGetUniformLocation(shader_program, "lightDir")
    light_color_loc = glGetUniformLocation(shader_program, "lightColor")
    object_color_loc = glGetUniformLocation(shader_program, "objectColor")
    #view_pos_loc = glGetUniformLocation(shader_program, "viewPos")

    # uniformのセット
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniform3f(light_dir_loc, 0.0, -3.0, -1.0)  # ディレクショナルライトの方向例
    glUniform3f(light_color_loc, 0.7, 0.7, 0.7)  # 白色光
    glUniform3f(object_color_loc, 1.0, 0.5, 0.31) # オブジェクトの色
    #glUniform3f(view_pos_loc, 0.0, 0.0, 3.0)     # カメラ位置例

    # Draw cube
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    glfw.swap_buffers(window)

def main():
    global counter, state, lasttime
    global bodies, geoms

    # Initialize GLFW
    glfw.init()

    # Create Window
    window = glfw.create_window(800, 600, "PyOpenGL GLFW Cube", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)

    # Enable depth test for 3D rendering
    glEnable(GL_DEPTH_TEST)

    shader_program = create_shader_program()

    # Generate buffers and arrays
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    #物理演算とシェーダのループ部分
    while not glfw.window_should_close(window):
        
        use_shader_in_tutorial3(window, shader_program, VAO, VBO, EBO)#シェーダーで描画
        
        t = dt - (time.time() - lasttime)
        if (t > 0):
            time.sleep(t)
        
        counter += 1

        if state==0:
            if counter==20:
                drop_object()
            if objcount==30:
                state=1
                counter=0
        # State 1: Explosion and pulling back the objects
        elif state==1:
            if counter==100:
                explosion()
            if counter>300:
                pull()
            if counter==500:
                counter=20

        ##衝突検出部分を書き換え。#############
        # Simulate
        n = 4
        for i in range(n):
            for g1 in geoms:    
                for g2 in geoms:
        
                    near_callback((world,contactgroup), g1, g2)

            for g1 in geoms:

                near_callback((world,contactgroup), g1, floor)
                #space.collide((world,contactgroup), ode.collide_callback(g1, floor))
                # Simulation step
            world.step(dt/n)
            # Remove all contact joints
            contactgroup.empty()
        ##衝突検出部分を書き換え。終了。#############

        lasttime = time.time()
        
    # Cleanup
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteBuffers(1, [EBO])

    glfw.terminate()

if __name__ == "__main__":
    main()