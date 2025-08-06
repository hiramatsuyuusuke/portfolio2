#ODE-0.16.4のtutorial3.pyを書き換えたコードです。
#https://hiramatsuyuusuke.github.io/portfolio2/product1.html

import sys, os, random, time
from math import *
from OpenGL.GL import *
import glfw
import glm
import numpy as np

import ode

from PIL import Image
from PIL import ImageOps
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.models as models

# Vertex Shader
vertex_shader_source = """
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in float color_opacity;
layout(location = 4) in vec2 vertexUV;
layout(location = 5) in float vertexUV_flag;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;
out vec3 Color;
out float Color_opacity;
out vec2 UV;
out float UV_flag;

void main()
{
    FragPos = vec3(model * vec4(position, 1.0));
    Normal = mat3(transpose(inverse(model))) * normal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
    Color = color;
    Color_opacity = color_opacity;
    UV = vertexUV;
    UV_flag = vertexUV_flag;
}
"""

# Fragment Shader
fragment_shader_source = """
#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec3 Color;
in float Color_opacity;
in vec2 UV; // UV座標
in float UV_flag;

out vec4 FragColor;

uniform vec3 lightDir;    // 光源の方向（正規化済み）
uniform vec3 lightColor;
uniform sampler2D texture0;
uniform sampler2D texture1;

void main()
{
    // 法線ベクトルの正規化
    vec3 norm = normalize(Normal);

    // 光源方向は逆ベクトルで受け取ることが多いのでadjustしてください
    vec3 lightDirection = normalize(-lightDir);

    // ディフューズ光：光の方向と法線の内積
    float diff = max(dot(norm, lightDirection), 0.0);

    // 環境光のみ簡易実装
    vec3 ambient = 0.7 * lightColor;

    // 拡散光強度分だけ色乗算
    vec3 diffuse = diff * lightColor;

    //テクスチャの切り替え
    if (UV_flag<0.5)
    {
        vec3 result = (ambient + diffuse) * Color;
        FragColor = vec4(result, Color_opacity);
    }
    else
    {
        if (UV_flag<0.8)
        {
            FragColor = texture2D(texture0, UV);
        }
        else
        {
            FragColor = texture2D(texture1, UV);
        }
    }
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
def draw_body(body, vertices, indices, color_rgba):
    """Draw an ODE body.
    """
    ##################################################################
    #boxの頂点データを作成
    if body.shape == "box":
        #カラーの設定
        color_r, color_g, color_b, color_a = color_rgba    

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
            arr2 = np.array([   vpx[i]+px, vpy[i]+py, vpz[i]+pz,        # positions
                                nx[i], ny[i], nz[i],                    # normals
                                color_r, color_g, color_b, color_a,     # color
                                1.0, 1.0, 0.0],                         #uv and flag
                                dtype=np.float32)
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

    ##################################################################
    #cylinderの頂点データを作成。シリンダーオブジェクトの座標は重心座標。
    if body.shape == "cylinder":
        color_r, color_g, color_b, color_a = color_rgba     #カラー
        rad_num = 10    #円周方向の分割数
        height_num = 2    #高さ方向の分割数
        drad = 360/rad_num  #円周方向の刻み幅
        r, h = body.cylindersize    #
        v = []  #頂点座標のリスト
        #回転前のcylinderの頂点座標の計算
        for hn in range(height_num + 1):    #高さ方向
            for rn in range(rad_num):       #円周方向
                angle = radians( rn * drad )
                cylinder_vx = r * cos(angle)
                cylinder_vy = r * sin(angle)
                cylinder_vz = ( h/float(height_num) )*hn - 0.5*h    #オブジェクト座標を重心にする
                #回転前の頂点座標データをリストに追加
                v.append( glm.vec3(cylinder_vx, cylinder_vy, cylinder_vz) ) 
        v.append( glm.vec3(0, 0, 0.5 * h) ) #上天板の中心
        v.append( glm.vec3(0, 0, -0.5 * h) ) #下天板の中心

        #回転前の頂点の法線データ
        n = []  #法線データのリスト
        for hn in range(height_num + 1):    #高さ方向
            for i in range(rad_num):        #円周方向
                angle = radians( i * drad )
                cylinder_vx = cos(angle)
                cylinder_vy = sin(angle)
                if hn == 0:
                    cylinder_vz = 0 # -1    #一番下の法線のz成分
                elif hn == height_num:
                    cylinder_vz = 0 # 1     #一番上の法線のz成分
                else:
                    cylinder_vz = 0         #
                #回転前の頂点の法線データ
                n.append( glm.vec3(cylinder_vx, cylinder_vy, cylinder_vz) )
        n.append( glm.vec3(0, 0, 1) ) #上天板の中心
        n.append( glm.vec3(0, 0, -1) ) #下天板の中心

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
        for i in range(rad_num*(height_num + 1)+2): # +2は上下の天板の分
            rotated_vector = quaternion * v[i]    # 頂点座標を回転
            x,y,z = rotated_vector #回転後の頂点座標
            vpx.append(x)
            vpy.append(y)
            vpz.append(z)

        #頂点の法線の回転変換
        nx = []
        ny = []
        nz = []    
        for i in range(rad_num*(height_num + 1)+2): # +2は上下の天板の分
            rotated_vector = quaternion * n[i]    # 頂点の法線を回転
            x,y,z = glm.normalize(rotated_vector)
            nx.append(x)
            ny.append(y)
            nz.append(z)
            
        # cylinder vertices and normals (position XYZ + normals)
        px,py,pz = body.getPosition()   #cylinderの座標を取得
        arr1 = np.array([], dtype=np.float32)
        for i in range(rad_num*(height_num + 1)+2): # +2は上下の天板の分
            arr2 = np.array([   vpx[i]+px, vpy[i]+py, vpz[i]+pz,        # positions
                                nx[i], ny[i], nz[i],                    # normals
                                color_r, color_g, color_b, color_a,     # color
                                1.0, 1.0, 0.0],                         #uv and flag
                                dtype=np.float32)
            arr1 = np.append(arr1, arr2)
        vertices_result = np.append(vertices, arr1)
        
        # Indices defining the triangles composing the cylinder
        i_start = max(indices) + 1  #天板用のindicesを記録しておく
        #cylinder側面のindices
        for hn in range(height_num):    #高さ方向
            #indicesの続きの値を取得
            if len(indices) == 0:
                i = 0
            else:
                if hn == 0:
                    i = max(indices) + 1
                else:
                    i = max(indices) + 1 - rad_num  #一段下（一つ前の上側）のindicesを基準にして、上側に新しいindicesを数える。
            #indicesの計算
            arr3 = np.array( [], dtype=np.uint32 ) #一周分のindicesのリスト
            for s in range( rad_num ):  #円周方向
                if s < rad_num - 1:
                    #cylinder側面のindices
                    arr4 = np.array([   s + 0 + i,
                                        s + 1 + i,
                                        s + 1 + rad_num + i,

                                        s + 1 + rad_num + i,
                                        s + rad_num + i,
                                        s + 0 + i ], dtype=np.uint32)
                else:
                    #円周方向のindicesのつなぎ目
                    arr4 = np.array([ rad_num - 1 + i,
                                        0 + i,
                                        rad_num + i,

                                        rad_num + i, 
                                        rad_num * 2 -1 + i,
                                        rad_num - 1 + i ], dtype=np.uint32)
                #indicesをリストに追加      
                arr3 = np.append(arr3, arr4)
            #一周分のindicesをリストに追加     
            indices = np.append(indices, arr3)
        
        #indicesの続きの値を取得
        if len(indices) == 0:
            i = 0
        else:
            i = max(indices) + 1
        #cylinder天板のindices    
        for s in range( rad_num ):  #円周方向
            if s < rad_num - 1:
                #cylinder天板のindices
                arr4 = np.array([ i,        #天板の中心のindices
                                  s + 0 + i - rad_num,  #側面の最後の一周のindeices
                                  s + 1 + i - rad_num,  #側面の最後の一周のindeices

                                  1 + i,    #天板の中心のindices
                                  s + 0 + i_start,  #側面の最初の一周のindeices
                                  s + 1 + i_start   #側面の最初の一周のindeices
                                  ], dtype=np.uint32)
            else:   #円周方向のつなぎ目
                arr4 = np.array([ i,        #天板の中心のindices
                                  s + 0 + i - rad_num,            #側面の最後の一周のindeices
                                  s + 1 + i - rad_num - rad_num,  #側面の最後の一周のindeices

                                  1 + i,    #天板の中心のindices
                                  s + 0 + i_start,              #側面の最初の一周のindeices
                                  s + 1 + i_start - rad_num     #側面の最初の一周のindeices
                                  ], dtype=np.uint32)
            #indicesをリストに追加
            arr3 = np.append(arr3, arr4)
        #一周分のindicesをリストに追加  
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

# create_cylinder
def create_cylinder(world, space, density, direction, r, h):
    """Create a cylinder body and its corresponding geom."""
    # Create body
    body = ode.Body(world)
    M = ode.Mass()
    M.setCylinder(density, direction, r, h)
    body.setMass(M)

    # Set parameters for drawing the body
    body.shape = "cylinder"
    body.cylindersize = (r, h)
    
    # Create a box geom for collision detection
    geom = ode.GeomCylinder(space, r, h)
    geom.setBody(body)
    return body, geom

# drop_object
def drop_object():
    """Drop an object into the scene."""

    global bodies, geoms, counter, objcount

    body, geom = create_box(world, space, 1000, 1.0,0.2,0.2)
    body.setPosition( (0.0, 3.0, 0.0) )
    theta = 0.0
    ct = cos (theta)
    st = sin (theta)
    body.setRotation([ct, 0., -st, 0., 1., 0., st, 0., ct])
    bodies.append(body)
    geoms.append(geom)
    #counter=0
    objcount+=1

# drop_box_robo
def drop_box_robo( init_object_i, density):
    """Drop an object into the scene."""
    global bodies, geoms, objcount

    lx, ly, lz = init_object_i[1]
    px, py, pz = init_object_i[2]

    body, geom = create_box(world, space, density, lx, ly, lz)
    theta = 0
    body.setPosition( (px, py, pz) )
    ct = cos (theta)
    st = sin (theta)
    body.setRotation([ct, 0., -st, 0., 1., 0., st, 0., ct])#y軸回転
    #body.setRotation([1., 0., 0., 0., ct, -st, 0., st, ct])#x軸回転 
    bodies_robo.append(body)
    geoms_robo.append(geom)
    objcount += 1

# drop_box_object
def drop_box( init_object_i, density):
    """Drop an object into the scene."""
    global bodies, geoms, objcount

    lx, ly, lz = init_object_i[1]
    px, py, pz = init_object_i[2]

    body, geom = create_box(world, space, density, lx, ly, lz)
    theta = 0
    body.setPosition( (px, py, pz) )
    ct = cos (theta)
    st = sin (theta)
    body.setRotation([ct, 0., -st, 0., 1., 0., st, 0., ct])#y軸回転
    #body.setRotation([1., 0., 0., 0., ct, -st, 0., st, ct])#x軸回転 
    bodies.append(body)
    geoms.append(geom)
    objcount += 1

# drop_cylinder_object
def drop_cylinder( init_object_i, density):
    """Drop an object into the scene."""
    global bodies, geoms, objcount

    r,h = init_object_i[1]
    px, py, pz = init_object_i[2]

    #odeとopenglのシリンダーの方向を一致させるために、3(z軸方向)にする。
    body, geom = create_cylinder(world, space, density, 3, r, h)  
    if init_object_i[0] == "cylinder_z":    #init_object_i= [ name, shape, position, color ]
        theta = 3.1415*(0.0/180.0)  #シリンダーの方向はz軸方向
    if init_object_i[0] == "cylinder_y":    #init_object_i= [ name, shape, position, color ]
        theta = 3.1415*(90.0/180.0) #シリンダーの方向をy軸方向にして、シリンダーを立てる。
    body.setPosition( (px, py, pz) )  
    ct = cos (theta)
    st = sin (theta)
    body.setRotation([1., 0., 0., 0., ct, st, 0., -st, ct])#x軸回転
    bodies.append(body)
    geoms.append(geom)
    objcount += 1

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

#画面キャプチャ
def capture2():
    width = 320
    height = 320

    glReadBuffer(GL_FRONT)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    data = glReadPixels(340, 100, width, height, GL_RGB, GL_UNSIGNED_BYTE)

    image = Image.frombytes("RGB", (width, height), data)
    image = ImageOps.flip(image)
    
    new_size = (32, 32) # 縮小するサイズを指定 (幅, 高さ)
    resized_image = image.resize(new_size)  # 画像をリサイズ
    #resized_image.save( "capture2()_test.jpg" )  # 縮小した画像を保存

    return resized_image

#ResNet18を使って、衝突するかどうかを判定
def display_image_recognition():

    # 画面をキャプチャ
    field_image = capture2()

    # 画像をTensorに変換
    transform = transforms.ToTensor()
    tensor_image = transform(field_image)
    input_tensor = tensor_image.unsqueeze(0)  # バッチ次元を追加

    # 推論
    with torch.no_grad():
        Learned_model_output = Learned_model(input_tensor)

    #衝突するかどうかを判定
    if Learned_model_output[0,0] < Learned_model_output[0,1]:
        print(1)
        return 1
    else:
        print(0)
        return 0

#テクスチャ読み込み関数
def load_texture(texture_file_name):

    # Generate a texture ID
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id )

    img = Image.open(texture_file_name)
    w, h = img.size
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.tobytes())
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)    # Set texture parameters
    glBindTexture(GL_TEXTURE_2D, 0)  # Unbind texture

    return texture_id

#4つの方角（北東南西）の処理が終わったときに衝突判定の学習用クラスラベルデータを出力
def box_collision_class_label_file(class_label, txt_file_name):
    with open(txt_file_name,"w") as o:
        for index, v in enumerate(class_label):
            if index == len(class_label) - 1:
                print(str(v), end="", file=o)
            else:
                print(str(v) + ",", end="", file=o)

#無効座標ファイル作成用
def invalid_position_matrix_file(invalid_position_matrix, txt_file_name):
    with open(txt_file_name,"w") as o:
        # ループで全要素を取得
        for row in invalid_position_matrix:
            for index, element in enumerate(row):
                if index == len(row) - 1:
                    print(str(element), file=o)  
                else:                                  
                    print(str(element) + ",", end=" ", file=o)


######################################################################
# Initialize GLFW
glfw.init()
if not glfw.init():
    glfw.terminate()

# Create Window
window = glfw.create_window(680, 450, "学習用データの作成。左:俯瞰画像。右:視界画像。", None, None)
if not window:
    glfw.terminate()
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
bodies_robo = []

# The geoms for each of the bodies
geoms = []
geoms_robo = []

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

#roboオブジェクト描画の初期設定
init_object_robo = []
                         #name          #shape                  #position                       #color
init_object_robo.append(["box_robo",    ( 0.4, 0.4, 0.4),     (  0.0, 0.2,  0.0 ),           ( 0.5, 0.9, 0.1, 1.0)])     #0 : 箱ロボット

#オブジェクト描画の初期設定
init_object = []
                    #name               #shape                  #position                       #color
init_object.append(["box",              ( 1.30, 0.2, 1.70),     (  3.8,  0.7,  0.82),           ( 0.4, 0.23, 0.2, 1.0)])    #0 : 机の天板
init_object.append(["box",              ( 0.15, 0.6, 0.15),     (  3.3,  0.3,  0.00),           ( 0.4, 0.23, 0.2, 1.0)])    #1 : 机の脚
init_object.append(["box",              ( 0.15, 0.6, 0.15),     (  4.3,  0.3,  0.00),           ( 0.4, 0.23, 0.2, 1.0)])    #2 : 机の脚
init_object.append(["box",              ( 1.00, 0.6, 0.50),     (  3.8,  0.3,  1.40),           ( 0.4, 0.23, 0.2, 1.0)])    #3 : 机の引き出し
init_object.append(["box",              ( 1.00, 0.1, 2.00),     ( -3.7,  0.6,  1.00),           ( 0.2, 0.20, 0.2, 1.0)])    #4 : 棚の天板1
init_object.append(["box",              ( 1.00, 0.1, 2.00),     ( -3.7,  1.0,  1.00),           ( 0.2, 0.20, 0.2, 1.0)])    #5 : 棚の天板2
init_object.append(["box",              ( 0.15, 1.0, 0.15),     ( -4.2,  0.5,  0.00),           ( 0.2, 0.20, 0.2, 1.0)])    #6 : 棚の脚
init_object.append(["box",              ( 0.15, 1.0, 0.15),     ( -3.2,  0.5,  0.00),           ( 0.2, 0.20, 0.2, 1.0)])    #7 : 棚の脚
init_object.append(["box",              ( 0.15, 1.0, 0.15),     ( -3.2,  0.5,  2.00),           ( 0.2, 0.20, 0.2, 1.0)])    #8 : 棚の脚
init_object.append(["box",              ( 0.15, 1.0, 0.15),     ( -4.2,  0.5,  2.00),           ( 0.2, 0.20, 0.2, 1.0)])    #9 : 棚の脚
init_object.append(["cylinder_y",       ( 0.4, 0.1),            (  0.0,  0.05, 2.0 ),           ( 1.0, 1.0, 1.0, 1.0)])     #10 : 扇風機の足1
init_object.append(["cylinder_y",       ( 0.1, 0.6),            (  0.0,  0.3,  2.0 ),           ( 1.0, 1.0, 1.0, 1.0)])     #11 : 扇風機の足2
init_object.append(["cylinder_z",       ( 0.3, 0.1),            (  0.0,  0.7,  1.75),           ( 1.0, 1.0, 1.0, 1.0)])     #12 : 扇風機の頭1
init_object.append(["cylinder_z",       ( 0.1, 0.4),            (  0.0,  0.7,  2.0 ),           ( 1.0, 1.0, 1.0, 1.0)])     #13 : 扇風機の頭2
init_object.append(["box",              ( 2.2,  0.2, 1.20),     (  1.0,  0.3, -3.5 ),           ( 0.5, 0.5, 0.5, 1.0)])     #14 : #ベッドの天板
init_object.append(["cylinder_y",       ( 0.1, 0.2),            (  0.0,  0.1, -4.0 ),           ( 0.5, 0.5, 0.5, 1.0)])     #15 : #ベッドの足
init_object.append(["cylinder_y",       ( 0.1, 0.2),            (  2.0,  0.1, -4.0 ),           ( 0.5, 0.5, 0.5, 1.0)])     #16 : #ベッドの足
init_object.append(["cylinder_y",       ( 0.1, 0.2),            (  2.0,  0.1, -3.0 ),           ( 0.5, 0.5, 0.5, 1.0)])     #17 : #ベッドの足
init_object.append(["cylinder_y",       ( 0.1, 0.2),            (  0.0,  0.1, -3.0 ),           ( 0.5, 0.5, 0.5, 1.0)])     #18 : #ベッドの足
init_object.append(["box",              ( 0.3, 1.0, 8.99),     ( -4.651, 0.50,  0.0),           ( 0.4, 0.23, 0.2, 1.0)])    #19 : 外周左
init_object.append(["box",              ( 0.3, 1.0, 8.99),     (  4.651, 0.50,  0.0),           ( 0.4, 0.23, 0.2, 1.0)])    #20 : 外周右
init_object.append(["box",              ( 8.99, 1.0, 0.3),     (  0.0, 0.50,  4.651),           ( 0.4, 0.23, 0.2, 1.0)])    #21 : 外周手前
init_object.append(["box",              ( 8.99, 1.0, 0.3),     (  0.0, 0.50, -4.651),           ( 0.4, 0.23, 0.2, 1.0)])    #22 : 外周奥

#roboオブジェクトをbodies_robo[]とgeoms_robo[]に入れる
drop_box_robo(init_object_robo[0], 1.0)

#オブジェクトをbodies[]とgeoms[]に入れる
for iob in init_object: # iob  = [ name, shape, position, color ]
    # cylinder
    if iob[0] == "cylinder_z" or iob[0] == "cylinder_y":    # iob[0]はname
        drop_cylinder(iob, 10)  #(init_object[i], density)
    # box
    else:
        drop_box(iob, 1.0) #drop_box_robo(init_object[i], density)

#固定ジョイントの作成
fixed_joints=[]
#障害物オブジェクトの固定ジョイントの作成
for index, b in enumerate(bodies):
    fixed_joints.append(ode.FixedJoint(world))
    fixed_joints[len(fixed_joints)-1].attach(b, None)  #障害物オブジェクトをその場に固定
    fixed_joints[len(fixed_joints)-1].setFixed()


#シェーダーで描画
def use_shader_in_tutorial3(window, shader_program, VAO, VBO, EBO, gaze_x, gaze_z):

    vertices = np.array([], dtype=np.float32)
    indices = np.array([], dtype=np.uint32)

    #床と壁の頂点データを作成
    #床と壁のverticesデータを作成
                           # positions        # normals         # color               # texture_uv and flag
    floor_arr = np.array([  4.5, 0.0,  4.5,   0.0, 1.0,  0.0,   1.0, 1.0, 1.0, 1.0,   4.5, 4.5, 0.7,   #床 0
                            4.5, 0.0, -4.5,   0.0, 1.0,  0.0,   1.0, 1.0, 1.0, 1.0,   4.5, 0.0, 0.7,   #床 1
                           -4.5, 0.0,  4.5,   0.0, 1.0,  0.0,   1.0, 1.0, 1.0, 1.0,   0.0, 4.5, 0.7,   #床 2
                           -4.5, 0.0, -4.5,   0.0, 1.0,  0.0,   1.0, 1.0, 1.0, 1.0,   0.0, 0.0, 0.7,   #床 3

                           -4.5, 0.0, -4.5,   0.0, 0.0,  1.0,   1.0, 1.0, 1.0, 1.0,   0.0, 0.0, 1.0,   #壁奥 4
                            4.5, 0.0, -4.5,   0.0, 0.0,  1.0,   1.0, 1.0, 1.0, 1.0,   4.5, 0.0, 1.0,   #壁奥 5
                           -4.5, 2.0, -4.5,   0.0, 0.0,  1.0,   1.0, 1.0, 1.0, 1.0,   0.0, 1.0, 1.0,   #壁奥 6
                            4.5, 2.0, -4.5,   0.0, 0.0,  1.0,   1.0, 1.0, 1.0, 1.0,   4.5, 1.0, 1.0,   #壁奥 7

                           -4.5, 0.0, -4.5,   1.0, 0.0,  0.0,   1.0, 1.0, 1.0, 1.0,   0.0, 0.0, 1.0,   #壁左 8
                           -4.5, 0.0,  4.5,   1.0, 0.0,  0.0,   1.0, 1.0, 1.0, 1.0,   4.5, 0.0, 1.0,   #壁左 9
                           -4.5, 2.0, -4.5,   1.0, 0.0,  0.0,   1.0, 1.0, 1.0, 1.0,   0.0, 1.0, 1.0,   #壁左 10
                           -4.5, 2.0,  4.5,   1.0, 0.0,  0.0,   1.0, 1.0, 1.0, 1.0,   4.5, 1.0, 1.0,   #壁左 11

                            4.5, 0.0, -4.5,  -1.0, 0.0,  0.0,   1.0, 1.0, 1.0, 1.0,   0.0, 0.0, 1.0,   #壁右 12
                            4.5, 0.0,  4.5,  -1.0, 0.0,  0.0,   1.0, 1.0, 1.0, 1.0,   4.5, 0.0, 1.0,   #壁右 13
                            4.5, 2.0, -4.5,  -1.0, 0.0,  0.0,   1.0, 1.0, 1.0, 1.0,   0.0, 1.0, 1.0,   #壁右 14
                            4.5, 2.0,  4.5,  -1.0, 0.0,  0.0,   1.0, 1.0, 1.0, 1.0,   4.5, 1.0, 1.0,   #壁右 15

                           -4.5, 0.0,  4.5,   0.0, 0.0, -1.0,   1.0, 1.0, 1.0, 1.0,   0.0, 0.0, 1.0,   #壁手前 16
                            4.5, 0.0,  4.5,   0.0, 0.0, -1.0,   1.0, 1.0, 1.0, 1.0,   4.5, 0.0, 1.0,   #壁手前 17
                           -4.5, 2.0,  4.5,   0.0, 0.0, -1.0,   1.0, 1.0, 1.0, 1.0,   0.0, 1.0, 1.0,   #壁手前 18
                            4.5, 2.0,  4.5,   0.0, 0.0, -1.0,   1.0, 1.0, 1.0, 1.0,   4.5, 1.0, 1.0    #壁手前 19
                           ], dtype=np.float32)
    vertices = np.append(vertices, floor_arr)
    #床と壁のIndicesデータを作成
    if len(indices) == 0:
        i = 0
    else:
        i = max(indices) + 1
    arr3 = np.array([0+i,1+i,2+i,  3+i,1+i,2+i,  #床
                     4+i,5+i,6+i,  7+i,5+i,6+i,  #壁奥
                     8+i,9+i,10+i,  11+i,9+i,10+i,    #壁左
                     12+i,13+i,14+i,  15+i,13+i,14+i, #壁右
                     16+i,17+i,18+i,  19+i,17+i,18+i  #壁手前    
                     ], dtype=np.uint32)
    indices = np.append(indices, arr3)

    #roboオブジェクトの頂点データを作成
    vertices, indices = draw_body(bodies_robo[0], vertices, indices, init_object_robo[0][3])  # init_object[index][3]は、colorのデータ

    #odeオブジェクトの頂点データを作成
    for index, b in enumerate(bodies):
        #bodyの頂点データを作成
        vertices, indices = draw_body(b, vertices, indices, init_object[index][3])  # init_object[index][3]は、colorのデータ

    glBindVertexArray(VAO)

    # Vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Element buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 13 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 13 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
    glEnableVertexAttribArray(1)

    # color attribute
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 13 * vertices.itemsize, ctypes.c_void_p(6 * vertices.itemsize))
    glEnableVertexAttribArray(2)

    # color opacity attribute
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 13 * vertices.itemsize, ctypes.c_void_p(9 * vertices.itemsize))
    glEnableVertexAttribArray(3)

    # uv attribute
    glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, 13 * vertices.itemsize, ctypes.c_void_p(10 * vertices.itemsize))
    glEnableVertexAttribArray(4)

    # uv flag attribute
    glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, 13 * vertices.itemsize, ctypes.c_void_p(12 * vertices.itemsize))
    glEnableVertexAttribArray(5)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    # Projection matrix (perspective)
    projection = np.identity(4, dtype=np.float32)

    fov = 60
    aspect_ratio = 320 / 320
    near = 0.5
    far = 30.0

    f = 1.0 / tan(radians(fov) / 2)
    projection[0, 0] = f / aspect_ratio
    projection[1, 1] = f
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = (2 * far * near) / (near - far)
    projection[3, 2] = -0.5
    projection[3, 3] = 0
        
    # Start render 
    glfw.poll_events()

    glUseProgram(shader_program)

    glUniform1i(glGetUniformLocation(shader_program, "texture0"), 0) # GL_TEXTURE0を渡す
    glUniform1i(glGetUniformLocation(shader_program, "texture1"), 1) # GL_TEXTURE1を渡す

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

    # uniformのセット
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniform3f(light_dir_loc, 1.0, -3.0, -1.0)  # ディレクショナルライトの方向例
    glUniform3f(light_color_loc, 1.0, 1.0, 1.0)  # 白色光

    # 1つ目のビューポートの設定（左）
    # View matrix (camera)
    glViewport(10, 100, 320, 320)   

    #glm.lookat()でカメラの設定
    cameraPos = glm.vec3(6.0, 7.2, 7.5,)
    targetPos = glm.vec3(-1.0, -1.0, 0.0)
    upVector = glm.vec3(0.0, 1.0, 0.0)    
    view_glm = glm.lookAt(cameraPos, targetPos, upVector)   #俯瞰の視点

    # numpyの回転行列にglmの回転行列の姿勢データを入れる
    view = np.identity(4, dtype=np.float32)
    view[0,0] = view_glm[0,0]
    view[1,0] = view_glm[1,0]
    view[2,0] = view_glm[2,0]
    view[3,0] = view_glm[3,0]

    view[0,1] = view_glm[0,1]
    view[1,1] = view_glm[1,1]
    view[2,1] = view_glm[2,1]
    view[3,1] = view_glm[3,1]

    view[0,2] = view_glm[0,2]
    view[1,2] = view_glm[1,2]
    view[2,2] = view_glm[2,2]
    view[3,2] = view_glm[3,2]
    
    view[0,3] = view_glm[0,3]
    view[1,3] = view_glm[1,3]
    view[2,3] = view_glm[2,3]
    view[3,3] = view_glm[3,3]

    # uniformのセット
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    # Draw cube
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    # 2つ目のビューポートの設定（右）
    # View matrix (camera)
    glViewport(340, 100, 320, 320)    

    robo_x, robo_y, robo_z = bodies_robo[0].getPosition()   #boxの座標を取得
    #gaze_x, gaze_y, gaze_z = [robo_x, 0.148, robo_z + 0.3]

    #glm.lookat()でカメラの設定
    cameraPos = glm.vec3( robo_x, 0.15, robo_z)
    targetPos = glm.vec3( robo_x + gaze_x, 0.148, robo_z + gaze_z)
    upVector = glm.vec3(0.0, 1.0, 0.0)    
    view_glm = glm.lookAt(cameraPos, targetPos, upVector)   #俯瞰の視点

    # numpyの回転行列にglm.lookAt()の回転行列のデータを入れる
    view = np.identity(4, dtype=np.float32)    
    view[0,0] = view_glm[0,0]
    view[1,0] = view_glm[1,0]
    view[2,0] = view_glm[2,0]
    view[3,0] = view_glm[3,0]

    view[0,1] = view_glm[0,1]
    view[1,1] = view_glm[1,1]
    view[2,1] = view_glm[2,1]
    view[3,1] = view_glm[3,1]

    view[0,2] = view_glm[0,2]
    view[1,2] = view_glm[1,2]
    view[2,2] = view_glm[2,2]
    view[3,2] = view_glm[3,2]
    
    view[0,3] = view_glm[0,3]
    view[1,3] = view_glm[1,3]
    view[2,3] = view_glm[2,3]
    view[3,3] = view_glm[3,3]

    # uniformのセット
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)    
    # Draw cube
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    glfw.swap_buffers(window)

def main():
    global counter, state, lasttime
    global bodies, geoms


    #counterイベント時間の設定
    timing_update_position_and_clear_box = 100
    timing_create_box = 102
    timing_judge1 = 107         #無効座標の判定。#timing_create_boxとtiming_judge1の間は、3カウント以上開ける。
    timing_img_capture = 109    #箱ロボットの視界をキャプチャ。
    timing_rolling_box = 111
    timing_judge2 = 131         #分類ラベルを判定。

    #箱ロボットと障害物の衝突判定用フラグ
    global_robo_obstacle_collision_flag = 0 

    #箱を転がす方角のカウント。
    rolling_direc_count = 0   #無効座標の取得=0、北=1、東=2、南=3、西=4
    rolling_direc_count_max = 4

    #箱ロボットの座標設定
    box_px_start = -4.0
    box_pz_start = -4.0
    box_px_end = 4.0
    box_pz_end = 4.0
    box_px = box_px_start  #箱のx座標の初期値をセット
    box_pz = box_pz_start  #箱のy座標の初期値をセット
    box_dpx = 0.25  #箱のx座標の探索の刻み幅
    box_dpz = 0.25  #箱のy座標の探索の刻み幅

    #箱ロボットの座標マトリックス設定
    ipmx = 0#無効座標ファイル作成用
    ipmz = 0#無効座標ファイル作成用
    ipms = int((abs(box_px_start)*2) / box_dpx) + 1 #無効座標マトリックスのサイズ

    #北=1
    gaze_x = 0 #視線方向の初期値をセット
    gaze_z = -0.3 #視線方向の初期値をセット
    Force_x = 0 #箱を押す力の方向の初期値をセット
    Force_z = -120  #箱を押す力の方向の初期値をセット


    #学習用のクラスラベル
    class_label = []  

    #最初から始めるとき
    if rolling_direc_count == 0:
        # ゼロ埋め2D配列を作成
        invalid_position_matrix = [[0 for _ in range(ipms)] for _ in range(ipms)]#無効座標ファイル作成用
    #途中から始めるとき
    if rolling_direc_count != 0:
        # Read lines from a file
        with open('invalid_position_matrix.txt', 'r') as file:
            lines = file.readlines()
        # Convert lines to a 2D NumPy array (assuming space-separated values)
        invalid_position_matrix = np.array([list(map(int, line.strip().split(","))) for line in lines])

    # 配列の省略表示を無効化
    #np.set_printoptions(threshold=np.inf)
    #print(invalid_position_matrix)



    #テクスチャ読み込み#
    tex_floor = load_texture("sample1.png")
    tex_wall = load_texture("sample2.png")
    # テクスチャ0にバインド
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex_floor)
    # テクスチャ1にバインド
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, tex_wall)

    #透明表現を有効にする
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Enable depth test for 3D rendering
    glEnable(GL_DEPTH_TEST)

    shader_program = create_shader_program()

    # Generate buffers and arrays
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    #物理演算とシェーダのループ部分
    while not glfw.window_should_close(window):

        #画面の背景をクリア
        glClearColor(0.1, 0.3, 0.3, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #ビューポートの範囲の背景色をクリア
        #Enable the scissor test
        glEnable(GL_SCISSOR_TEST)
        #Define the scissor box (x, y, width, height)
        glScissor(10, 100, 320, 320)
        glClearColor(0.15, 0.62, 0.89, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)     
        #Define the scissor box (x, y, width, height)
        glScissor(340, 100, 320, 320)
        glClearColor(0.15, 0.62, 0.89, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)   
        #Disable the scissor test when done
        glDisable(GL_SCISSOR_TEST)

        #シェーダーで描画
        use_shader_in_tutorial3(window, shader_program, VAO, VBO, EBO, gaze_x, gaze_z)


        #物理演算部分
        #t = dt - (time.time() - lasttime)
        #if (t > 0):
        #    time.sleep(t)
        
        if rolling_direc_count <= rolling_direc_count_max:#探索する方角の回数が4以下のとき。北東南西の４回。
            counter += 1

            #座標の更新
            if counter == timing_update_position_and_clear_box:
                box_px += box_dpx  #箱のx座標を更新
                ipmx += 1#無効座標ファイル作成用☆☆☆#
                #箱のx方向座標がmaxを超えたとき
                if box_px > box_px_end:
                    box_pz += box_dpz       #箱のz方向座標を更新
                    ipmz += 1#無効座標ファイル作成用☆☆☆#

                    #4つの方角（北東南西）の処理が終わったときに衝突判定の学習用クラスラベルデータを出力
                    if box_px > box_px_end and box_pz > box_pz_end and rolling_direc_count == rolling_direc_count_max:
                        if rolling_direc_count != 0:
                            box_collision_class_label_file( class_label, "label_data_for_learning_ResNet18.txt")
                            counter =  timing_update_position_and_clear_box - 1

                    #一つの方角の探索が終わったとき
                    if box_px > box_px_end and box_pz > box_pz_end:

                        if rolling_direc_count == 0:
                            #無効座標ファイル作成用☆☆☆#
                            invalid_position_matrix_file( invalid_position_matrix, "invalid_position_matrix.txt")

                        rolling_direc_count += 1 #rolling_direc_countを次の方角に更新
                        #次の方角の探索の始まりの座標
                        box_px = box_px_start
                        box_pz = box_pz_start
                        ipmx = 0#無効座標ファイル作成用☆☆☆#
                        ipmz = 0#無効座標ファイル作成用☆☆☆#
                        #e
                        if rolling_direc_count == 2:
                            gaze_x = 0.3
                            gaze_z = 0 
                            Force_x = 120 
                            Force_z = 0 
                        #s
                        if rolling_direc_count == 3:
                            gaze_x = 0                
                            gaze_z = 0.3
                            Force_x = 0
                            Force_z = 120
                        #w
                        if rolling_direc_count == 4:
                            gaze_x = -0.3
                            gaze_z = 0
                            Force_x = -120 
                            Force_z = 0

                        counter = timing_update_position_and_clear_box - 1

                    box_px = box_px_start     #箱のx方向座標を折り返し
                    ipmx = 0    #無効座標ファイル作成用

            #箱ロボットを作成     
            if counter == timing_create_box:
                #箱ロボットをクリア
                bodies_robo.clear()
                geoms_robo.clear()
                #無効座標を取得するとき
                if rolling_direc_count == 0:
                    #無効座標ファイル作成用に一回り大きい箱ロボットを作成する
                    #roboオブジェクトをbodies_robo[]とgeoms_robo[]に入れる
                                        #name        #shape            #position                 #color
                    init_object_robo[0]=["box_robo", ( 0.4, 0.4, 0.4), ( box_px, 0.2, box_pz ), ( 0.5, 0.9, 0.1, 1.0)]   #0 : 箱ロボット
                    drop_box_robo(init_object_robo[0], 1000.0)      #bodies_robo[0], geoms_robo[0]の作成
                #衝突判定をするとき
                else:
                    #箱ロボットを作成する。
                                        #name        #shape            #position                 #color
                    init_object_robo[0]=["box_robo", ( 0.3, 0.3, 0.3), ( box_px, 0.2, box_pz ), ( 0.5, 0.9, 0.1, 1.0)]   #0 : 箱ロボット
                    drop_box_robo(init_object_robo[0], 10.0)      #bodies_robo[0], geoms_robo[0]の作成
   
            #箱ロボットがセットされたとき、障害物と重なっているかどうかを判定。
            if counter == timing_judge1:
                #無効座標を取得するrolling_direc_countのとき、無効座標を取得する。
                if rolling_direc_count == 0:
                    #箱ロボットと障害物が重なったとき
                    if global_robo_obstacle_collision_flag == 1:
                        print("無効座標取得:"+str(ipmx)) 
                        invalid_position_matrix[ipmx][ipmz]= 3 #無効座標マトリックスの値を3にする
                        global_robo_obstacle_collision_flag = 0 #箱ロボットと障害物の衝突flagを0にして、衝突してない状態に戻す。  
                    #counterをカウンターを最初に戻す。無効座標取得時は、画像のキャプチャとラベルの取得は行わない。  
                    counter =  timing_update_position_and_clear_box - 1  #

                #衝突判定をするrolling_direc_countのとき、箱ロボットが作成された座標が無効座標なら、衝突判定をスキップする。
                if rolling_direc_count != 0:
                    #無効座標のとき、スキップする
                    if invalid_position_matrix[ipmx][ipmz] == 3:
                        print(3)
                        global_robo_obstacle_collision_flag = 0 #箱ロボットと障害物の衝突flagを0にして、衝突してない状態に戻す。 
                        counter = timing_update_position_and_clear_box - 1   #カウンターを最初に戻す。無効座標をスキップするときは、画像のキャプチャとラベルの取得は行わない。

            #学習用の画像のキャプチャ
            if counter == timing_img_capture:
                capture2().save( "img/test" + str(len(class_label)) + ".jpg")  # 縮小した画像を保存

            #箱を転がす1
            if counter == timing_rolling_box:
                bodies_robo[0].addForce(( Force_x, 0, Force_z))

            #箱を転がす2
            if counter == timing_rolling_box + 2:
                bodies_robo[0].addForce(( Force_x, 0, Force_z))        

            #衝突するか衝突しないかの学習用クラスラベルデータの取得
            if counter == timing_judge2: 
                #箱ロボットと障害物が衝突したとき、1のクラスラベルを取得
                if global_robo_obstacle_collision_flag == 1:
                    print(1)
                    class_label.append(1)
                    global_robo_obstacle_collision_flag = 0 #箱ロボットと障害物の衝突flagを0にして、衝突してない状態に戻す。
                #箱ロボットと障害物が衝突しなかったとき、0のクラスラベルを取得
                else:
                    print(0)
                    class_label.append(0)
                counter =  timing_update_position_and_clear_box - 1   #カウンターを最初に戻す

            #箱の作成が終わった後から、箱ロボットと障害物の衝突判定を開始する。判定するだけで、衝突処理は行わない。
            if timing_create_box + 1 < counter and counter < timing_judge2:
                #箱ロボットと障害物の衝突を検出し、フラグを立てる。
                for g1 in geoms:
                    for g2 in geoms_robo:
                        # Check if the objects do collide
                        contacts = ode.collide(g1, g2)
                        for c in contacts:
                            global_robo_obstacle_collision_flag = 1
                            
        ##衝突検出部分を書き換え。#############
        # Simulate
        n = 4
        for i in range(n):
            for g1 in geoms:
                near_callback((world,contactgroup), g1, floor)

            near_callback((world,contactgroup), geoms_robo[0], floor)

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

    # Unbind texture
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, 0)  # Unbind texture
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, 0)  # Unbind texture

    glfw.terminate()

if __name__ == "__main__":

    #学習していないResNet18 modelを読み込んでインスタンスを生成
    Learned_model = models.resnet18(weights = None)
    # Modify the final fully connected layer for a custom number of classes
    num_classes = 2 #衝突するクラスと衝突しないクラスの2つ
    Learned_model.fc = nn.Linear(Learned_model.fc.in_features, num_classes)
    #3D空間の画像を使って学習したResNet18の読み込み
    Learned_model.load_state_dict(torch.load("Weight1.pth", weights_only=True))
    Learned_model.eval()  # 推論モードに切り替え

    main()
