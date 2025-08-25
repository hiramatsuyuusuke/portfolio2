#ODE-0.16.4のtutorial3.pyを書き換えたコードです。
#https://hiramatsuyuusuke.github.io/portfolio2/product3.html

import random, time
from math import *
from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
import glfw
import numpy as np
from pyrr import Matrix44, Vector3

import ode


#深度マップ生成用シェーダー
# Vertex Shader
depth_vertex_shader_source = """
#version 330 core

layout(location = 0) in vec3 position;

uniform mat4 lightSpaceMatrix;
uniform mat4 model;

void main()
{
    model;  //ここで一回modelを実行しないと深度マップが正常に機能しない。理由は不明。
    gl_Position = lightSpaceMatrix * model * vec4(position, 1.0);
}
"""
# Fragment Shader
depth_fragment_shader_source = """
#version 330 core

void main() {
    // 深度値のみを出力
}
"""

#シーン描画用シェーダー
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
    out vec3 Normal;
    out vec4 FragPosLightSpace;
} vs_out;

void main()
{
    vs_out.Normal = mat3(transpose(inverse(model))) * normal;
    vs_out.FragPosLightSpace = lightSpaceMatrix * model * vec4(position, 1.0);

    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""
# Fragment Shader
fragment_shader_source = """
#version 330 core
in VS_OUT {
    in vec3 Normal;
    in vec4 FragPosLightSpace;
} fs_in;

uniform sampler2D shadowMap;
uniform vec3 lightDir;    // 光源の方向（正規化済み）
uniform vec3 lightColor;
uniform vec3 objectColor;

float ShadowCalculation(vec4 fragPosLightSpace)
{
    // プロジェクション分割(正規化)
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5; // [0,1]に変換
    
    // シャドウ判定（バイアス付け）
    //float bias = 0.005;
    float bias = 0.0005;    
    float currentDepth = projCoords.z;
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    //任意の範囲内で深度を比較した結果(0 or 1)を足し合わせる
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {    
            // 深度をシャドウマップから取得
            float closestDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > closestDepth? 1.0 : 0.0;
        }
    }
    shadow /= 9.0; //深度を比較した任意の範囲で割る

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

#depthシェーダプログラム作成
def create_depth_shader_program():
    depth_vertex_shader = compile_shader(depth_vertex_shader_source, GL_VERTEX_SHADER)
    depth_fragment_shader = compile_shader(depth_fragment_shader_source, GL_FRAGMENT_SHADER)
    depth_program = glCreateProgram()    
    glAttachShader(depth_program, depth_vertex_shader)
    glAttachShader(depth_program, depth_fragment_shader)
    glLinkProgram(depth_program)
    # Check linking
    if not glGetProgramiv(depth_program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(depth_program).decode()
        raise RuntimeError(f"Program linking failed: {error}")
    glDeleteShader(depth_vertex_shader)
    glDeleteShader(depth_fragment_shader)
    return depth_program

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
    v.append( (-box_vx, -box_vy, -box_vz) )   #0
    v.append( ( box_vx, -box_vy, -box_vz) )   #1
    v.append( ( box_vx,  box_vy, -box_vz) )   #2
    v.append( (-box_vx,  box_vy, -box_vz) )   #3
    # front face
    v.append( (-box_vx, -box_vy,  box_vz) )   #4
    v.append( ( box_vx, -box_vy,  box_vz) )   #5
    v.append( ( box_vx,  box_vy,  box_vz) )   #6
    v.append( (-box_vx,  box_vy,  box_vz) )   #7
    # bottom face
    v.append( (-box_vx, -box_vy,  box_vz) )   #8
    v.append( ( box_vx, -box_vy,  box_vz) )   #9
    v.append( ( box_vx, -box_vy, -box_vz) )   #10
    v.append( (-box_vx, -box_vy, -box_vz) )   #11
    # top face
    v.append( ( box_vx,  box_vy,  box_vz) )   #12
    v.append( (-box_vx,  box_vy,  box_vz) )   #13
    v.append( (-box_vx,  box_vy, -box_vz) )   #14
    v.append( ( box_vx,  box_vy, -box_vz) )   #15
    # left face
    v.append( (-box_vx, -box_vy,  box_vz) )   #16
    v.append( (-box_vx,  box_vy,  box_vz) )   #17
    v.append( (-box_vx,  box_vy, -box_vz) )   #18
    v.append( (-box_vx, -box_vy, -box_vz) )   #19
    # right face
    v.append( ( box_vx, -box_vy,  box_vz) )   #20
    v.append( ( box_vx,  box_vy,  box_vz) )   #21
    v.append( ( box_vx,  box_vy, -box_vz) )   #22
    v.append( ( box_vx, -box_vy, -box_vz) )   #23

   #回転前の頂点の法線データ（面方向の法線）
    n = []
    for i in range(24):
        if i < 4:
            n.append((  0,  0, -1)) # back face   # 0,1,2,3
        elif i < 8:
            n.append((  0,  0,  1))  # front face   # 4,5,6,7
        elif i < 12:
            n.append((  0, -1,  0))  # bottom face   # 8,9,10,11
        elif i < 16:
            n.append((  0,  1,  0))  # top face   # 12,13,14,15
        elif i < 20:
            n.append(( -1,  0,  0))  # left face   # 16,17,18,19
        elif i < 24:
            n.append((  1,  0,  0))  # right face   # 20,21,22,23

    # Cube vertices and normals (position XYZ + normals)
    arr1 = np.array([], dtype=np.float32)
    for i in range(24):
                          # vertex positions          # normals
        arr2 = np.array([ v[i][0], v[i][1], v[i][2],  n[i][0], n[i][1], n[i][2]], dtype=np.float32)
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
def use_shader_in_tutorial3(shader_program, depth_shader_program, VAO, VBO, EBO, FBO, depth_texture, vertices_data_list):

    #
    glUseProgram(depth_shader_program)

    glClearColor(0.2, 0.3, 0.3, 1)
    glBindFramebuffer(GL_FRAMEBUFFER, FBO)  #FBOにバインド
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    #一つのオブジェクトごとに頂点データを転送して描画。深度マップを生成するためにFBOにレンダリング。
    for vdl in vertices_data_list:

        #オブジェクトごとの頂点リストから頂点データを取得
        vertices, indices, body_index = vdl

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
        #
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # Model matrix: rotate box
        model = np.identity(4, dtype=np.float32)
        #boxの姿勢を取得して回転行列に代入
        if body_index != None:
            R = bodies[body_index].getRotation()
            model[0,0] = R[0]
            model[1,0] = R[1]
            model[2,0] = R[2]
            model[0,1] = R[3]
            model[1,1] = R[4]
            model[2,1] = R[5]
            model[0,2] = R[6]
            model[1,2] = R[7]
            model[2,2] = R[8]
        #boxの座標を取得して回転行列に代入
        if body_index != None:
            px,py,pz = bodies[body_index].getPosition()
            model[3, 0] = px
            model[3, 1] = py
            model[3, 2] = pz

        #レンダリングの設定：ライト視点の深度マップを生成するためのレンダリング。
        # Define light's position and target
        light_position = Vector3([0.0, 2.0*3.0, 3.0*3.0])
        light_target = Vector3([0.0, 0.0, 0.0])
        light_up = Vector3([0.0, 1.0, 0.0])
        # Create light's view matrix (lookAt matrix)
        light_view = Matrix44.look_at(light_position, light_target, light_up)
        # Define light's projection matrix
        light_projection = Matrix44.perspective_projection(45.0, 800.0/600.0, 1, 20)
        # Combine to form the lightSpaceMatrix
        lightSpaceMatrix = light_projection * light_view

        # Set uniform matrices
        model_loc = glGetUniformLocation(shader_program, "model")
        light_space_matrix_loc = glGetUniformLocation(shader_program, "lightSpaceMatrix")

        # uniformのセット
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(light_space_matrix_loc, 1, GL_FALSE, lightSpaceMatrix)

        # Draw cube. 1回目のレンダリング：ライト視点の深度マップを生成するためのレンダリング。
        glViewport(0, 0, 800, 600)
        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    #
    glUseProgram(shader_program)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)  #デフォルトフレームバッファにバインド
    # デフォルトフレームバッフをクリア
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    #一つのオブジェクトごとに頂点データを転送して描画。シャドウマッピングで影のあるシーンをレンダリング。
    for vdl in vertices_data_list:

        #boxごとの頂点リストから頂点データを取得
        vertices, indices, body_index = vdl

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

        # Model matrix: rotate box
        model = np.identity(4, dtype=np.float32)
        #boxの姿勢を取得して回転行列に代入
        if body_index != None:
            R = bodies[body_index].getRotation()
            model[0,0] = R[0]
            model[1,0] = R[1]
            model[2,0] = R[2]
            model[0,1] = R[3]
            model[1,1] = R[4]
            model[2,1] = R[5]
            model[0,2] = R[6]
            model[1,2] = R[7]
            model[2,2] = R[8]
        #boxの座標を取得して回転行列に代入
        if body_index != None:
            px,py,pz = bodies[body_index].getPosition()
            model[3, 0] = px
            model[3, 1] = py
            model[3, 2] = pz

        #レンダリングの設定。シーンのレンダリング。
        # Define view position and target
        view_position = Vector3([1.0, 4.0, 4.0])
        view_target = Vector3([0.0, 0.0, 0.0])
        view_up = Vector3([0.0, 1.0, 0.0])
        # Create view matrix (lookAt matrix)
        view = Matrix44.look_at(view_position, view_target, view_up)

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
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, light_projection)
        glUniformMatrix4fv(light_space_matrix_loc, 1, GL_FALSE, lightSpaceMatrix)
        glUniform3f(light_dir_loc, 0.0, -2.0, -3.0)  # ディレクショナルライトの方向例
        glUniform3f(light_color_loc, 0.7, 0.7, 0.7)  # 白色光
        glUniform3f(object_color_loc, 1.0, 0.5, 0.31) # オブジェクトの色
        glUniform1i(depth_map_location, 0)  # テクスチャユニット0を指定 

        # Draw cube. シャドウマッピングで影のあるシーンをレンダリング。
        glViewport(0, 0, 800, 600)
        glBindTexture(GL_TEXTURE_2D, depth_texture )# テクスチャ0（ライト視点の深度マップ）にバインド。depth_textureとFBOはアタッチされている。
        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        glBindTexture(GL_TEXTURE_2D, 0)  # Unbind texture
        

def main():
    global counter, state, lasttime
    global bodies, geoms

    # Initialize GLFW
    glfw.init()

    # Create Window
    window = glfw.create_window(800, 600, "PyOpenGL GLFW drop box", None, None)
    if not window:
        glfw.terminate()
        return

    #
    glfw.make_context_current(window)

    #シャドウマッピング用にフレームバッファと深度テクスチャを設定。
    # Create a framebuffer
    FBO = glGenFramebuffers(1)  #ウィンドウ生成直後にフレームバッファを生成するのが無難
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
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)
    #デフォルトフレームバッファ（画面のバッファ）にバインド
    glBindFramebuffer(GL_FRAMEBUFFER, 0) 
    
    #
    depth_shader_program = create_depth_shader_program()
    shader_program = create_shader_program()

    # Generate buffers and arrays
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    #オブジェクトごとの頂点データリスト（シェーダー用のリスト）
    vertices_data_list = [] #[ [vertices, indices, ode object number], ... ]

    #床のverticesデータを作成
                                # positions         # normals        
    floor_vertices = np.array([ -3.0,  0.0,  -3.0,   0.0, 1.0,  0.0,  #床 0
                                -3.0,  0.0,   3.0,   0.0, 1.0,  0.0,  #床 1
                                 3.0,  0.0,  -3.0,   0.0, 1.0,  0.0,  #床 2
                                 3.0,  0.0,   3.0,   0.0, 1.0,  0.0   #床 3
                                ], dtype=np.float32)
    #床のIndicesデータを作成
    floor_indices = np.array([0,1,2,  3,1,2  #床
                    ], dtype=np.uint32)
    #オブジェクトごとの頂点データリストに床の頂点データを追加
    vertices_data_list.append([floor_vertices, floor_indices, None])
    #オブジェクトごとの頂点データリスト（シェーダー用のリスト）の要素数 = 床の数 = 1
    init_obj_num = len(vertices_data_list)

    #物理演算とシェーダのループ部分
    while not glfw.window_should_close(window):

        #頂点データリスト（シェーダー用のリスト）にbody（ODEのオブジェクト）の頂点データを追加
        for index, body in enumerate(bodies):
            #「body（ODEのオブジェクト）の数」が、「オブジェクトごとの頂点データリスト（シェーダー用のリスト）の要素数-床の数」より多い時、
            # 頂点データリストにbodyの頂点データを追加
            if index + 1 > len(vertices_data_list) - init_obj_num:
                #
                vertices = np.array([], dtype=np.float32)
                indices = np.array([], dtype=np.uint32)
                #body（ODEのオブジェクト）の頂点データ（シェーダー用）を作成
                vertices, indices = draw_body(body, vertices, indices)
                #頂点データリスト（シェーダー用のリスト）にbody（ODEのオブジェクト）の頂点データを追加
                vertices_data_list.append([vertices, indices, index])

        #シェーダーで描画
        use_shader_in_tutorial3(shader_program, depth_shader_program, VAO, VBO, EBO, FBO, depth_texture, vertices_data_list)  

        #物理演算
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
    glDeleteTextures([depth_texture])
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteBuffers(1, [EBO])
    glDeleteBuffers(1, [FBO])
    glfw.terminate()


if __name__ == "__main__":

    main()