#ODE-0.16.4のtutorial3.pyを書き換えたコードです。
#https://hiramatsuyuusuke.github.io/portfolio2/product3.html

import sys, os, random, time
from math import *
from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
import glfw
import glm
import numpy as np

import ode

from pyrr import Matrix44, Vector3


# Vertex Shader
vertex_shader_source = """
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;

out VS_OUT {
    out vec3 FragPos;
    out vec3 Normal;
    out vec4 FragPosLightSpace;
} vs_out;

void main()
{
    vs_out.FragPos = vec3(model * vec4(position, 1.0));
    vs_out.Normal = mat3(transpose(inverse(model))) * normal;
    vs_out.FragPosLightSpace = lightSpaceMatrix * vec4(vs_out.FragPos, 1.0);

    gl_Position = projection * view * vec4(vs_out.FragPos, 1.0);
}
"""

# Fragment Shader
fragment_shader_source = """
#version 330 core
in VS_OUT {
    in vec3 FragPos;
    in vec3 Normal;
    in vec4 FragPosLightSpace;
} fs_in;

uniform sampler2D shadowMap;
uniform vec3 lightDir;    // 光源の方向（正規化済み）
uniform vec3 lightColor;
uniform vec3 objectColor;
uniform vec3 viewPos;

float ShadowCalculation(vec4 fragPosLightSpace)
{
    // プロジェクション分割(正規化)
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5; // [0,1]に変換
    
    // 深度をシャドウマップから取得
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    float currentDepth = projCoords.z;

    // シャドウ判定（バイアス付け）
    float bias = 0.005;
    float shadow = currentDepth - bias > closestDepth? 1.0 : 0.0;

    // シャドウ範囲外は影無しにする
    if(projCoords.z > 1.0)
        shadow = 0.0;
    return shadow;
}

void main()
{
    //
    float shadow = ShadowCalculation(fs_in.FragPosLightSpace);

    // 法線ベクトルの正規化
    vec3 norm = normalize(fs_in.Normal);

    // 光源方向は逆ベクトルで受け取ることが多いのでadjustしてください
    vec3 lightDirection = normalize(-lightDir);

    // ディフューズ光：光の方向と法線の内積
    float diff = max(dot(norm, lightDirection), 0.0);

    // 環境光のみ簡易実装
    vec3 ambient = 0.5 * lightColor;

    // 拡散光強度分だけ色乗算
    vec3 diffuse = diff * lightColor;

    //
    vec3 result = ((1.0 - shadow * 0.7) * (diffuse) + ambient) * objectColor;
    gl_FragColor = vec4(result, 1.0);
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
def draw_body(body, vertices, indices):
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
    v.append( glm.vec3(-box_vx, -box_vy, -box_vz) )    #0
    v.append( glm.vec3( box_vx, -box_vy, -box_vz) )   #1
    v.append( glm.vec3( box_vx,  box_vy, -box_vz) )   #2
    v.append( glm.vec3(-box_vx,  box_vy, -box_vz) )   #3
    # front face
    v.append( glm.vec3(-box_vx, -box_vy,  box_vz) )   #4
    v.append( glm.vec3( box_vx, -box_vy,  box_vz) )   #5
    v.append( glm.vec3( box_vx,  box_vy,  box_vz) )   #6
    v.append( glm.vec3(-box_vx,  box_vy,  box_vz) )   #7
    # bottom face
    v.append( glm.vec3(-box_vx, -box_vy,  box_vz) )   #8
    v.append( glm.vec3( box_vx, -box_vy,  box_vz) )   #9
    v.append( glm.vec3( box_vx, -box_vy, -box_vz) )   #10
    v.append( glm.vec3(-box_vx, -box_vy, -box_vz) )   #11
    # top face
    v.append( glm.vec3( box_vx,  box_vy,  box_vz) )   #12
    v.append( glm.vec3(-box_vx,  box_vy,  box_vz) )   #13
    v.append( glm.vec3(-box_vx,  box_vy, -box_vz) )   #14
    v.append( glm.vec3( box_vx,  box_vy, -box_vz) )   #15
    # left face
    v.append( glm.vec3(-box_vx, -box_vy,  box_vz) )   #16
    v.append( glm.vec3(-box_vx,  box_vy,  box_vz) )   #17
    v.append( glm.vec3(-box_vx,  box_vy, -box_vz) )   #18
    v.append( glm.vec3(-box_vx, -box_vy, -box_vz) )   #19
    # right face
    v.append( glm.vec3( box_vx, -box_vy,  box_vz) )   #20
    v.append( glm.vec3( box_vx,  box_vy,  box_vz) )   #21
    v.append( glm.vec3( box_vx,  box_vy, -box_vz) )   #22
    v.append( glm.vec3( box_vx, -box_vy, -box_vz) )   #23

   #回転前の頂点の法線データ（面方向の法線）
    n = []
    for i in range(24):
        if i < 4:
            n.append(glm.vec3(  0,  0, -1)) # back face   # 0,1,2,3
        elif i < 8:
            n.append(glm.vec3(  0,  0,  1))  # front face   # 4,5,6,7
        elif i < 12:
            n.append(glm.vec3(  0, -1,  0))  # bottom face   # 8,9,10,11
        elif i < 16:
            n.append(glm.vec3(  0,  1,  0))  # top face   # 12,13,14,15
        elif i < 20:
            n.append(glm.vec3( -1,  0,  0))  # left face   # 16,17,18,19
        elif i < 24:
            n.append(glm.vec3(  1,  0,  0))  # right face   # 20,21,22,23

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
    for i in range(24):
        rotated_vector = quaternion * v[i]    # 頂点座標を回転
        x,y,z = rotated_vector #回転後の頂点座標
        vpx.append(x)
        vpy.append(y)
        vpz.append(z)

    #頂点の法線の回転変換
    nx = []
    ny = []
    nz = []    
    for i in range(24):
        rotated_vector = quaternion * n[i]    # 頂点の法線を回転
        x,y,z = glm.normalize(rotated_vector)
        nx.append(x)
        ny.append(y)
        nz.append(z)
        
    # Cube vertices and normals (position XYZ + normals)
    px,py,pz = body.getPosition()   #boxの座標を取得
    arr1 = np.array([], dtype=np.float32)
    for i in range(24):
                         # positions                        # normals
        arr2 = np.array([ vpx[i]+px, vpy[i]+py, vpz[i]+pz,  nx[i], ny[i], nz[i]], dtype=np.float32)
        arr1 = np.append(arr1, arr2)
    vertices_result = np.append(vertices, arr1)
    
    # Indices defining the 12 triangles composing the cube
    if len(indices) == 0:
        i = 0
    else:
        i = max(indices) + 1
    arr3 = np.array([
        0+i,1+i,2+i, 2+i,3+i,0+i,  # back face
        4+i,5+i,6+i, 6+i,7+i,4+i,  # front face
        8+i,9+i,10+i, 10+i,11+i,8+i,  # bottom face
        12+i,13+i,14+i, 14+i,15+i,12+i,  # top face
        16+i,17+i,18+i, 18+i,19+i,16+i,  # left face
        20+i,21+i,22+i, 22+i,23+i,20+i   # right face
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

# Initialize GLFW
glfw.init()

# Create Window
window = glfw.create_window(800, 600, "PyOpenGL GLFW drop box", None, None)
if not window:
    glfw.terminate()
    #return

#
glfw.make_context_current(window)

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
def use_shader_in_tutorial3(shader_program, VAO, VBO, EBO, FBO, depth_texture):

    #頂点のデータ
    vertices = np.array([], dtype=np.float32)
    #三角形の頂点の番号
    indices = np.array([], dtype=np.uint32)

    #床と壁の頂点データを作成
    #床と壁のverticesデータを作成
                        # positions        # normals        
    floor_arr = np.array([  -3.0,  0.0,  -3.0,   0.0, 1.0,  0.0,  #床 0
                            -3.0,  0.0,   3.0,   0.0, 1.0,  0.0,  #床 1
                             3.0,  0.0,  -3.0,   0.0, 1.0,  0.0,  #床 2
                             3.0,  0.0,   3.0,   0.0, 1.0,  0.0  #床 3
                        ], dtype=np.float32)
    vertices = np.append(vertices, floor_arr)   #頂点のデータ
    #床と壁のIndicesデータを作成
    if len(indices) == 0:
        i = 0
    else:
        i = max(indices) + 1
    arr3 = np.array([0+i,1+i,2+i,  3+i,1+i,2+i,  #床
                    ], dtype=np.uint32)
    indices = np.append(indices, arr3)  #三角形の頂点の番号

    #bodyの頂点データを作成
    for b in bodies:
        #bodyの頂点データを作成
        vertices, indices = draw_body(b, vertices, indices)
    
    #
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

    #
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)


    # Start render 

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
    
    #1回目のレンダリングの設定：ライト視点の深度マップを生成するためのレンダリング。#####################
    # Define light's position and target
    light_position = Vector3([0.0, 3.0, 1.0])
    light_target = Vector3([0.0, 0.0, 0.0])
    light_up = Vector3([0.0, 1.0, 0.0])

    # Create light's view matrix (lookAt matrix)
    light_view = Matrix44.look_at(light_position, light_target, light_up)

    # Define light's projection matrix (orthographic for directional light)
    light_projection = Matrix44.orthogonal_projection(-3.0, 3.0, -3.0, 3.0, -10.0, 10.0)

    # Combine to form the lightSpaceMatrix
    lightSpaceMatrix = light_projection * light_view

    # Set uniform matrices
    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    proj_loc = glGetUniformLocation(shader_program, "projection")
    light_dir_loc = glGetUniformLocation(shader_program, "lightDir")
    light_color_loc = glGetUniformLocation(shader_program, "lightColor")
    object_color_loc = glGetUniformLocation(shader_program, "objectColor")
    light_space_matrix_loc = glGetUniformLocation(shader_program, "lightSpaceMatrix")
    depth_map_location = glGetUniformLocation(shader_program, "shadowMap")

    # uniformのセット
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, light_view)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, light_projection)
    glUniformMatrix4fv(light_space_matrix_loc, 1, GL_FALSE, lightSpaceMatrix)
    glUniform3f(light_dir_loc, 0.0, -3.0, -1.0)  # ディレクショナルライトの方向例
    glUniform3f(light_color_loc, 0.7, 0.7, 0.7)  # 白色光
    glUniform3f(object_color_loc, 1.0, 0.5, 0.31) # オブジェクトの色
    glUniform1i(depth_map_location, 0)  # テクスチャユニット0を指定 

    # Draw cube. 1回目のレンダリング：ライト視点の深度マップを生成するためのレンダリング。
    # デフォルトフレームバッファにレンダリングしてからFBOに転送しているので、オフスクリーンレンダリングには、なっていない。
    # オフスクリーンレンダリングには、レンダーバッファの生成とアタッチが必要？
    glViewport(0, 0, 800, 600)
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
    
    # デフォルトフレームバッファをFBOに転送。depth_textureとFBOはアタッチされている。
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)   #デフォルトフレームバッファにバインド
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, FBO)   #FBOにバインド
    glBlitFramebuffer(0, 0, 800, 600,
                        0, 0, 800, 600,
                        GL_DEPTH_BUFFER_BIT, GL_NEAREST)    #転送
    glBindFramebuffer(GL_FRAMEBUFFER, 0)  #デフォルトフレームバッファにバインド

    # デフォルトフレームバッフへの1回目のレンダリングをクリア
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    #2回目のレンダリングの設定#####################
    # Define view position and target
    view_position = Vector3([1.0, 0.5, 1.0])
    view_target = Vector3([0.0, 0.0, 0.0])
    view_up = Vector3([0.0, 1.0, 0.0])

    # Create view matrix (lookAt matrix)
    view = Matrix44.look_at(view_position, view_target, view_up)

    # uniformのセット
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    # Draw cube. 2回目のレンダリング。影付きのレンダリング。
    glViewport(0, 0, 800, 600)
    glBindTexture(GL_TEXTURE_2D, depth_texture )# テクスチャ0（ライト視点の深度マップ）にバインド。depth_textureとFBOはアタッチされている。
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
    glBindTexture(GL_TEXTURE_2D, 0)  # Unbind texture
    

def main():
    global counter, state, lasttime
    global bodies, geoms

    #シャドウマッピング用にフレームバッファと深度テクスチャを設定。
    # Create a framebuffer
    FBO = glGenFramebuffers(1)  #
    glEnable(GL_DEPTH_TEST)     # Enable depth test for 3D rendering
    glDepthMask(GL_TRUE)        #
    glDepthFunc( GL_LEQUAL )    #
    #深度テクスチャ分の領域の生成
    depth_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depth_texture )# テクスチャ0にバインド
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 800, 600, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None) #CPUを使わないようにするため、Noneを指定して転送しない
    # テクスチャパラメータの設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glBindTexture(GL_TEXTURE_2D, 0)  # Unbind texture
    #FBOにバインド
    glBindFramebuffer(GL_FRAMEBUFFER, FBO)  
    #深度テクスチャ分の領域をFBOフレームバッファにアタッチ
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture, 0)
    #デフォルトフレームバッファ（画面のバッファ）にバインド
    glBindFramebuffer(GL_FRAMEBUFFER, 0) 
    
    #
    shader_program = create_shader_program()

    # Generate buffers and arrays
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    #物理演算とシェーダのループ部分
    while not glfw.window_should_close(window):

        use_shader_in_tutorial3(shader_program, VAO, VBO, EBO, FBO, depth_texture)  #シェーダーで描画

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
        
        glfw.swap_buffers(window)
        glfw.poll_events()


    # Cleanup
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteBuffers(1, [EBO])
    glDeleteBuffers(1, [FBO])
    glfw.terminate()

if __name__ == "__main__":
    main()