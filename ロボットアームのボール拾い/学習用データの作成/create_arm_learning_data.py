#ODE-0.16.4のtutorial3.pyを書き換えたコードです。
#https://hiramatsuyuusuke.github.io/portfolio2/product2.html

import sys, os, random, time
from math import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import ode

from PIL import Image
from PIL import ImageOps
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.models as models


# geometric utility functions
def scalp (vec, scal):
    vec[0] *= scal
    vec[1] *= scal
    vec[2] *= scal

def length (vec):
    return sqrt (vec[0]**2 + vec[1]**2 + vec[2]**2)

# prepare_GL
def prepare_GL():
    """Prepare drawing.
    """
    # Viewport
    glViewport(0,0,320,320) #ビューポートの大きさを320×320にする

    # Initialize
    glClearColor(0,0,0,0)   #背景を黒にする
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glEnable(GL_LIGHTING)  
    glEnable(GL_NORMALIZE)
    glShadeModel(GL_FLAT)

    # Projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective (45,1.3333,0.2,20)

    # Initialize ModelView matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Light source
    glLightfv(GL_LIGHT0,GL_POSITION,[ 0.3, 1, 0.3, 0])  #照明の位置を変更
    glLightfv(GL_LIGHT0,GL_DIFFUSE,[1,1,1,1])
    glLightfv(GL_LIGHT0,GL_SPECULAR,[1,1,1,1])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5,0.5,0.5,0.5]) #環境光の設定
    glEnable(GL_LIGHT0)

    # View transformation
    #gluLookAt (2.4, 3.6, 4.8, 0.5, 0.5, 0, 0, 1, 0)    #視点変換をコメントアウト

# draw_body
def draw_body(body):
    """Draw an ODE body.
    """

    x,y,z = body.getPosition()
    R = body.getRotation()
    rot = [R[0], R[3], R[6], 0.,
           R[1], R[4], R[7], 0.,
           R[2], R[5], R[8], 0.,
           x, y, z, 1.0]
    glPushMatrix()
    glMultMatrixd(rot)
    if body.shape=="box":
        sx,sy,sz = body.boxsize
        glScalef(sx, sy, sz)
        glutSolidCube(1)

    if body.shape=="sphere":
        r = body.spheresize
        quadric = gluNewQuadric()
        gluSphere(quadric, r, 10, 10)
        gluDeleteQuadric(quadric)

    if body.shape=="cylinder":
        #ODEのcylinder（重心座標）とOpenGLのcylinder（ボトム座標）の座標設定の違いを修整して描画する。

        #シリンダーの外周を作成
        r,h = body.cylindersize
        R=[1., 0., 0., 0., 1, 0, 0., 0, 1]#x軸周りに0度回転
        rot = [R[0], R[3], R[6], 0.,
            R[1], R[4], R[7], 0.,
            R[2], R[5], R[8], 0.,
            0, 0, -h*0.5, 1.0]  #z軸方向に -h*0.5 平行移動
        glMultMatrixd(rot)
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluCylinder(quadric, r, r, h, 10, 10)  # Base radius, top radius, height, slices, stacks    
        gluDeleteQuadric(quadric)
        
        #シリンダーの上下2枚のディスクを作成
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluDisk(quadric, 0, r, 10, 10)  # Inner radius, outer radius, slices, loops
        rot = [R[0], R[3], R[6], 0.,
            R[1], R[4], R[7], 0.,
            R[2], R[5], R[8], 0.,
            0, 0, h, 1.0]   #z軸方向に h 平行移動

        glMultMatrixd(rot)
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluDisk(quadric, 0, r, 10, 10)  # Inner radius, outer radius, slices, loops
        gluDeleteQuadric(quadric)

    glPopMatrix()

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

# create_sphere
def create_sphere(world, space, density, r):
    """Create a sphere body and its corresponding geom."""

    # Create body
    body = ode.Body(world)
    M = ode.Mass()
    M.setSphere(density, r)
    body.setMass(M)

    # Set parameters for drawing the body
    body.shape = "sphere"
    body.spheresize = r

    # Create a Sphere geom for collision detection
    geom = ode.GeomSphere(space, r)
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

# drop__box_robo_object
def drop_box_robo( lx, ly, lz, px, py, pz, density, y_rotation):
    """Drop an object into the scene."""
    global bodies_robo, geoms_robo, objcount

    body, geom = create_box(world, space, density, lx, ly, lz)
    body.setPosition( (px, py, pz) )
    theta = y_rotation
    ct = cos (theta)
    st = sin (theta)
    body.setRotation([ct, 0., -st, 0., 1., 0., st, 0., ct])#y軸回転
    bodies_robo.append(body)
    geoms_robo.append(geom)
    objcount += 1

# drop_box_object
def drop_box( lx, ly, lz, px, py, pz, density):
    """Drop an object into the scene."""
    global bodies, geoms, objcount

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

# drop_sphere_object
def drop_sphere( r, px, py, pz, density):
    """Drop an object into the scene."""
    global bodies, geoms, objcount

    body, geom = create_sphere(world, space, density, r)
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
def drop_cylinder( rotation_num, r, h, px, py, pz, density):
    """Drop an object into the scene."""
    global bodies, geoms, objcount
    #odeとopenglのシリンダーの方向を一致させるために、3(z軸方向)にする。
    body, geom = create_cylinder(world, space, density, 3, r, h)  
    
    if rotation_num == 1:
        theta = 3.1415*(0.0/180.0)  #シリンダーの方向はz軸方向
    if rotation_num == 2:
        theta = 3.1415*(90.0/180.0) #シリンダーの方向をy軸方向にして、シリンダーを立てる。

    body.setPosition( (px, py, pz) )  
    ct = cos (theta)
    st = sin (theta)
    body.setRotation([1., 0., 0., 0., ct, st, 0., -st, ct])#x軸回転
    #body.setRotation([ct, 0., -st, 0., 1., 0., st, 0., ct])#y軸回転
    #body.setRotation([ct, st., 0., -st, ct, 0., 0., 0., 1.])#z軸回転
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
        c.setBounce(0.01)
        c.setMu(5000)
        j = ode.ContactJoint(world, contactgroup, c)
        j.attach(geom1.getBody(), geom2.getBody())

#最初の一回だけウィンドウに描画
def draw1():
    glutSetWindow(winnum)

    glViewport(0,0,680,450) 
    glClearColor(0.2, 0.2, 0.2, 0.0)   
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glRasterPos2f(-0.95, -0.6)  # 描画位置を指定
    text = "The view overlooking the room."
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))  # 文字を描画   

    glRasterPos2f(-0.95, -0.78)  # 描画位置を指定
    text = "The obstacles are recognized using "
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))  # 文字を描画   

    glRasterPos2f(-0.95, -0.89)  # 描画位置を指定
    text = "the view from the green box robot."
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))  # 文字を描画   


    glRasterPos2f(0.05, -0.6)  # 描画位置を指定
    text = "The view from the green box robot."
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))  # 文字を描画   

    glRasterPos2f(0.05, -0.78)  # 描画位置を指定
    text = "These images are being input to the "
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))  # 文字を描画   

    glRasterPos2f(0.05, -0.89)  # 描画位置を指定
    text = "learned model ( ResNet18 )."
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))  # 文字を描画   

    glFlush()
    glutSwapBuffers()
 
#最初の一回だけサブウィンドウ0に描画
def draw2():
    glutSetWindow(subwinnum[0])

    glClearColor(0.0, 1.0, 0.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glFlush()
    glutSwapBuffers()

#最初の一回だけサブウィンドウ1に描画
def draw3():
    glutSetWindow(subwinnum[1])

    glClearColor(0.0, 1.0, 1.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glFlush()
    glutSwapBuffers()

#画面キャプチャ
def capture2():
    width = glutGet(GLUT_WINDOW_WIDTH)
    height = glutGet(GLUT_WINDOW_HEIGHT)

    glReadBuffer(GL_FRONT)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

    image = Image.frombytes("RGB", (width, height), data)
    image = ImageOps.flip(image)
    
    new_size = (32, 32) # 縮小するサイズを指定 (幅, 高さ)
    resized_image = image.resize(new_size)  # 画像をリサイズ

    return resized_image

#ResNet18を使って、アームを下降させるかどうかを判定
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

    #アームを下降させるかどうかを判定
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

#テクスチャポリゴン
def draw_tex_polygon():
    #glClear(GL_COLOR_BUFFER_BIT)#prepare_GL()の中で実行される

    glMaterialfv(GL_FRONT, GL_AMBIENT, [0.5, 0.5, 0.5, 0.5])    #環境光の影響  
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])    #地の色の設定

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)#テクスチャの色と地の色の混ざり方の設定
    glEnable(GL_BLEND)
    glEnable(GL_TEXTURE_2D)
    glNormal3f(0, 1, 0)     #glNormal3f()は非推奨の関数らしい

    #床
    glBindTexture(GL_TEXTURE_2D, tex_floor)
    glBegin(GL_QUADS)

    glTexCoord2d(4.5, 0.0)
    glVertex3d(4.5, -0.01, -4.5)

    glTexCoord2d(0.0, 0.0)
    glVertex3d( -4.5, -0.01, -4.5)

    glTexCoord2d(0.0, 4.5)
    glVertex3d( -4.5, -0.01, 4.5)

    glTexCoord2d(4.5, 4.5)
    glVertex3d(4.5, -0.01, 4.5)

    glEnd()

    #壁　奥
    glBindTexture(GL_TEXTURE_2D, tex_wall)
    glBegin(GL_QUADS)

    glTexCoord2d(1.0, 0.0)
    glVertex3d(4.5,  0.0, -4.5)

    glTexCoord2d(0.0, 0.0)
    glVertex3d( -4.5, 0.0, -4.5)

    glTexCoord2d(0.0, 1.0)
    glVertex3d( -4.5, 2.0, -4.5)

    glTexCoord2d(1.0, 1.0)
    glVertex3d(4.5,  2.0, -4.5)

    glEnd()

    #壁　左
    glBegin(GL_QUADS)

    glTexCoord2d(1.0, 0.0)
    glVertex3d(-4.5, 0.0, 4.5)

    glTexCoord2d(0.0, 0.0)
    glVertex3d( -4.5, 0.0, -4.5)

    glTexCoord2d(0.0, 1.0)
    glVertex3d( -4.5, 2.0, -4.5)

    glTexCoord2d(1.0, 1.0)
    glVertex3d(-4.5,  2.0, 4.5)

    glEnd()

    #壁　手前
    glBegin(GL_QUADS)

    glTexCoord2d(1.0, 0.0)
    glVertex3d(4.5,  0.0, 4.5)

    glTexCoord2d(0.0, 0.0)
    glVertex3d( -4.5, 0.0, 4.5)

    glTexCoord2d(0.0, 1.0)
    glVertex3d( -4.5, 2.0, 4.5)

    glTexCoord2d(1.0, 1.0)
    glVertex3d(4.5,  2.0, 4.5)

    glEnd()

    #壁　右
    glBegin(GL_QUADS)

    glTexCoord2d(1.0, 0.0)
    glVertex3d(4.5, 0.0, 4.5)

    glTexCoord2d(0.0, 0.0)
    glVertex3d(4.5, 0.0, -4.5)

    glTexCoord2d(0.0, 1.0)
    glVertex3d(4.5, 2.0, -4.5)

    glTexCoord2d(1.0, 1.0)
    glVertex3d(4.5, 2.0, 4.5)

    glEnd()

    glDisable(GL_TEXTURE_2D)

    glDisable(GL_BLEND)

    glFlush()#??#

######################################################################

# Initialize Glut
glutInit ([])

# Open a window
glutInitDisplayMode (GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE)

x = 0
y = 0
width = 680
height = 450
glutInitWindowPosition (x, y);
glutInitWindowSize (width, height);
winnum = glutCreateWindow (b"testode")   #bを追加してbyte列にする。#
glutDisplayFunc(draw1)
subwinnum = []
glutInitWindowSize ( 320, 320);
subwinnum.append(glutCreateSubWindow(winnum, 10, 10, 320, 320))
glutDisplayFunc(draw2)
glutInitWindowSize ( 320, 320);
subwinnum.append(glutCreateSubWindow(winnum, 340, 10, 320, 320))
glutDisplayFunc(draw3)

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

grip_counter = 0    #ハンドを握る時間のカウンター
downswing_position = -0.9   #ハンドを下降させるx座標
drop_sphere_position = -1.0 #ボールをドロップする位置
global_pick_up_balls_flag = 1   #ボールを拾うタイミングと離すタイミングのフラグ
rise_flag = 0       #アームの上昇終了判定フラグ
pick_up_success_flag = 0    #ボールを運べたかどうかのフラグ

#学習用のクラスラベル
class_label = []  

#テクスチャ読み込み#
glutSetWindow(subwinnum[0])
tex_floor = load_texture("sample1.png")
tex_wall = load_texture("sample2.png")
glutSetWindow(subwinnum[1])
tex_floor = load_texture("sample1.png")
tex_wall = load_texture("sample2.png")

#ロボットアームの作成
#0:台座
drop_box_robo(0.3, 0.5, 0.3, 0.0, 0.25, 0.0, 1.0, 0.0)  #(lx, ly, lz, px, py, pz, density, y_rotation)
fixed_joint1 = ode.FixedJoint(world)
fixed_joint1.attach(bodies_robo[0], None)  # ボディを固定
fixed_joint1.setFixed()
#1:腕1
drop_box_robo(0.9, 0.1, 0.1, -0.7, 0.5, 0.0, 1.0, 0.0)  #(lx, ly, lz, px, py, pz, density, y_rotation)
hinge_joint1 = ode.HingeJoint(world)
hinge_joint1.attach(bodies_robo[0], bodies_robo[1])  # 物体1と物体2を接続
hinge_joint1.setAnchor((-0.25, 0.5, 0.0))  # Set the anchor point (hinge location)
hinge_joint1.setAxis((0, 0, 1))    # 回転軸を設定
#2:腕2
drop_box_robo(0.9, 0.1, 0.1, -1.8, 0.5, 0.0, 1.0, 0.0)  #(lx, ly, lz, px, py, pz, density, y_rotation)
hinge_joint2 = ode.HingeJoint(world)
hinge_joint2.attach(bodies_robo[1], bodies_robo[2])  # 物体1と物体2を接続
hinge_joint2.setAnchor((-1.25, 0.51, 0.0))  # Set the anchor point (hinge location)
hinge_joint2.setAxis((0, 0, 1))    # 回転軸を設定
#3:手首
drop_box_robo(0.01, 0.1, 0.3, -2.25, 0.5, 0.0, 1.0, 0.0)  #(lx, ly, lz, px, py, pz, density, y_rotation)
hinge_joint3 = ode.HingeJoint(world)
hinge_joint3.attach(bodies_robo[2], bodies_robo[3])  # 物体1と物体2を接続
hinge_joint3.setAnchor((-2.25, 0.51, 0.0))  # Set the anchor point (hinge location)
hinge_joint3.setAxis((0, 0, 1))    # 回転軸を設定
#4:手先
drop_box_robo(0.1, 0.1, 0.01, -2.3, 0.5, 0.15, 1.0, 0.0)  #(lx, ly, lz, px, py, pz, density, y_rotation)
fixed_joint3 = ode.FixedJoint(world)
fixed_joint3.attach(bodies_robo[3], bodies_robo[4])  # ボディを固定
fixed_joint3.setFixed()
#5:手先
drop_box_robo(0.1, 0.1, 0.01, -2.3, 0.5, -0.15, 1.0, 0.0)  #(lx, ly, lz, px, py, pz, density, y_rotation)
fixed_joint4 = ode.FixedJoint(world)
fixed_joint4.attach(bodies_robo[3], bodies_robo[5])  # ボディを固定
fixed_joint4.setFixed()
#6:手先4の指の根元
drop_box_robo(0.02, 0.13, 0.01, -2.37, 0.5, 0.15, 0.1, -0.0)  #(lx, ly, lz, px, py, pz, density, y_rotation)
hinge_joint5 = ode.HingeJoint(world)
hinge_joint5.attach(bodies_robo[4], bodies_robo[6])  # ボディを固定
hinge_joint5.setAnchor((-2.35, 0.5, 0.15))  # Set the anchor point (hinge location)
hinge_joint5.setAxis((0, 1, 0))    # 回転軸を設定
#7:手先4の内側の指。左奥。
drop_box_robo(0.01, 0.02, 0.1, -2.42, 0.54, 0.125, 0.1, -1.2)  #(lx, ly, lz, px, py, pz, density, y_rotation)
fixed_joint5 = ode.FixedJoint(world)
fixed_joint5.attach(bodies_robo[6], bodies_robo[7])  # ボディを固定
fixed_joint5.setFixed()
#8:手先4の内側の指。左手前。
drop_box_robo(0.01, 0.02, 0.1, -2.42, 0.46, 0.125, 0.1, -1.2)  #(lx, ly, lz, px, py, pz, density, y_rotation)
fixed_joint6 = ode.FixedJoint(world)
fixed_joint6.attach(bodies_robo[6], bodies_robo[8])  # ボディを固定
fixed_joint6.setFixed()
#9:手先5の指の根元
drop_box_robo(0.02, 0.13, 0.01, -2.37, 0.5, -0.15, 0.1, 0.0)  #(lx, ly, lz, px, py, pz, density, y_rotation)
hinge_joint6 = ode.HingeJoint(world)
hinge_joint6.attach(bodies_robo[5], bodies_robo[9])  # ボディを固定
hinge_joint6.setAnchor((-2.35, 0.5, -0.15))  # Set the anchor point (hinge location)
hinge_joint6.setAxis((0, 1, 0))    # 回転軸を設定
#10:手先5の内側の指。右奥。
drop_box_robo(0.01, 0.02, 0.1, -2.42, 0.54, -0.125, 0.1, 1.2)  #(lx, ly, lz, px, py, pz, density, y_rotation)
fixed_joint7 = ode.FixedJoint(world)
fixed_joint7.attach(bodies_robo[9], bodies_robo[10])  # ボディを固定
fixed_joint7.setFixed()
#11:手先5の内側の指。右手前。
drop_box_robo(0.01, 0.02, 0.1, -2.42, 0.46, -0.125, 0.1, 1.2)  #(lx, ly, lz, px, py, pz, density, y_rotation)
fixed_joint8 = ode.FixedJoint(world)
fixed_joint8.attach(bodies_robo[9], bodies_robo[11])  # ボディを固定
fixed_joint8.setFixed()

hinge_joint1.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
hinge_joint1.setParam(ode.ParamFMax, 1000)  # max力
hinge_joint2.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
hinge_joint2.setParam(ode.ParamFMax, 1000)  # max力

hinge_joint5.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
hinge_joint5.setParam(ode.ParamFMax, 1000)  # max力
hinge_joint6.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
hinge_joint6.setParam(ode.ParamFMax, 1000)  # max力

#囲い
drop_box_robo(0.05, 0.1, 1.0, -0.7, 0.05, 0.0, 1.0, 0.0)  #(lx, ly, lz, px, py, pz, density, y_rotation)
drop_box_robo(1.0, 0.1, 0.05, -0.1, 0.05, 0.5, 1.0, 0.0)  #(lx, ly, lz, px, py, pz, density, y_rotation)
drop_box_robo(0.05, 0.1, 1.0, 0.5, 0.05, 0.0, 1.0, 0.0)  #(lx, ly, lz, px, py, pz, density, y_rotation)
drop_box_robo(1.0, 0.1, 0.05, -0.1, 0.05, -0.5, 1.0, 0.0)  #(lx, ly, lz, px, py, pz, density, y_rotation)


# keyboard callback
def _keyfunc (c, x, y):
    sys.exit (0)

glutKeyboardFunc (_keyfunc)

# draw callback
def _drawfunc0 ():
    global bodies, bodies_robo
    # Draw the scene
    prepare_GL()

    #俯瞰の視点
    gluLookAt (3.0*2.0, 3.6*2.0, 5.0*1.5, -1.0, -1.0, 0, 0, 1, 0)#（視点位置、注視点位置、姿勢方向）

    for b in bodies:                       
        glMaterialfv(GL_FRONT, GL_AMBIENT, [1.0, 1.0, 1.0, 1.0])  #環境光の影響  
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        draw_body(b)

    for b in bodies_robo:           
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.5, 1, 0.5, 0.5])  #環境光の影響  
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 1, 0.5, 0.1])
        draw_body(b)

    draw_tex_polygon()

    glutSwapBuffers ()

def _drawfunc1 ():
    global bodies, objcount
    global bodies_robo
    # Draw the scene
    prepare_GL()
    
    x,y,z = bodies_robo[0].getPosition()#箱ロボットの座標

    #箱ロボットの視点
    gluLookAt ( x, y*2, z + 0.5, x - 1.0, y*2 - 0.1, z + 0.2, 0.0, 1.0, 0.0)#（視点位置、注視点位置、姿勢方向）

    for b in bodies:
        glMaterialfv(GL_FRONT, GL_AMBIENT, [1.0, 1.0, 1.0, 1.0])  #環境光の影響  
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        draw_body(b)

    for b in bodies_robo:
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.5, 1, 0.5, 0.5])  #環境光の影響  
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 1, 0.5, 0.1])
        draw_body(b)

    draw_tex_polygon()

    glutSwapBuffers ()
#glutDisplayFunc (_drawfunc)


# idle callback
def _idlefunc ():
    global counter, lasttime
    global bodies, geoms, subwinnum, world, contactgroup
    global bodies_robo, geoms_robo
    global grip_counter, downswing_position, global_pick_up_balls_flag
    global class_label, drop_sphere_position
    global rise_flag, pick_up_success_flag

    #t = dt - (time.time() - lasttime)
    #if (t > 0):
    #    time.sleep(t)

    if counter==0:
        bodies.clear()  #ボールをクリア
        geoms.clear()  #ボールをクリア
        rise_flag = 0
        #アームの下降の位置を戻す
        if downswing_position < -1.8:
            downswing_position = -0.9
            drop_sphere_position -= 0.1 #ボールの位置を更新

        downswing_position -= 0.01 #アームの下降の位置を更新
        global_pick_up_balls_flag = 1   #ボールを拾うタイミングと離すタイミングのフラグ
    #ボールを配置
    if counter==10:
        if drop_sphere_position <= -1.79 and drop_sphere_position >= -1.81: #-1.8
            drop_sphere( 0.07, 1.0, 0.1, 0.0, 0.1)  #(r, px, py, pz, density)
        
        elif drop_sphere_position <= -1.89 and drop_sphere_position >= -1.91: #-1.9
            drop_sphere( 0.07, -1.0, 0.1, 0.0, 0.1)  #(r, px, py, pz, density)
            drop_sphere( 0.07, -1.5, 0.1, 0.0, 0.1)  #(r, px, py, pz, density)     
        else:
            drop_sphere( 0.07, drop_sphere_position, 0.1, 0.0, 0.1)  #(r, px, py, pz, density)

    #手首の座標
    x,y,z = bodies_robo[3].getPosition() #手首の座標
    counter += 1

    #最初に、手首を曲げる
    if counter >10 and counter<=300:
        #最初以外はカウンターをとばす
        if grip_counter != 0:
            counter = 549
            grip_counter = 0
        #手首を曲げる
        hinge_joint3.setParam(ode.ParamVel, -0.3)  # 速度
        hinge_joint3.setParam(ode.ParamFMax, 200)  # max力   
        #手首をストップ
        if hinge_joint3.getAngle() < -1.57:
            hinge_joint3.setParam(ode.ParamVel, 0.0)  # 速度
            hinge_joint3.setParam(ode.ParamFMax, 200)  # max力   

    #アームの腕を曲げる。まっすぐ戻る。
    if counter==300:
        hinge_joint1.setParam(ode.ParamVel, 0.3)  # 速度
        hinge_joint1.setParam(ode.ParamFMax, 200)  # max力       
        hinge_joint2.setParam(ode.ParamVel, -0.6)  # 速度
        hinge_joint2.setParam(ode.ParamFMax, 200)  # max力
        hinge_joint3.setParam(ode.ParamVel, 0.3)  # 速度
        hinge_joint3.setParam(ode.ParamFMax, 200)  # max力     

    #アームの腕を上限まで曲げた状態で、根本の回転をストップ。
    if  counter < 550 and hinge_joint1.getAngle() > 1.45 :
        hinge_joint1.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
        hinge_joint1.setParam(ode.ParamFMax, 1000)  # max力 
        hinge_joint3.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
        hinge_joint3.setParam(ode.ParamFMax, 1000)  # max力 

    #アームの腕を上限まで曲げた状態で、肘の回転をストップ
    if counter < 550 and hinge_joint2.getAngle() < -2.90 :
        hinge_joint2.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
        hinge_joint2.setParam(ode.ParamFMax, 1000)  # max力 
    
    #アームの腕を伸ばす。直進。
    if counter == 550:
        hinge_joint1.setParam(ode.ParamVel, -0.3)  # 速度
        hinge_joint1.setParam(ode.ParamFMax, 200)  # max力
        hinge_joint2.setParam(ode.ParamVel, 0.6)  # 速度
        hinge_joint2.setParam(ode.ParamFMax, 200)  # max力
        hinge_joint3.setParam(ode.ParamVel, -0.3)  # 速度
        hinge_joint3.setParam(ode.ParamFMax, 200)  # max力 

    #アーム発進。
    if counter > 550:

        #下降
        if x < downswing_position and y > 0.25 and grip_counter == 0:
            #ボールを拾うタイミングと離すタイミングのフラグが1のとき
            if global_pick_up_balls_flag == 1: 
                global_pick_up_balls_flag = 2   #ボールを拾うタイミングと離すタイミングのフラグを2にする
                capture2().save( "img/test" + str(len(class_label)) + ".jpg")  # キャプチャ画像を縮小した画像を保存
            hinge_joint1.setParam(ode.ParamVel, -0.3)  # 速度
            hinge_joint1.setParam(ode.ParamFMax, 200)  # max力       
            hinge_joint2.setParam(ode.ParamVel, 0.0)  # 速度
            hinge_joint2.setParam(ode.ParamFMax, 200)  # max力
            hinge_joint3.setParam(ode.ParamVel, 0.3)  # 速度
            hinge_joint3.setParam(ode.ParamFMax, 200)  # max力  

        #下降をストップ   
        if y <= 0.25 and grip_counter < 200:
            #下降をストップ
            hinge_joint1.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
            hinge_joint1.setParam(ode.ParamFMax, 200)  # max力       
            hinge_joint2.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
            hinge_joint2.setParam(ode.ParamFMax, 200)  # max力
            hinge_joint3.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
            hinge_joint3.setParam(ode.ParamFMax, 200)  # max力                  
            grip_counter += 1

        #手先を握る
        if 1 <= grip_counter and grip_counter < 200:           
            #手先を握る
            hinge_joint5.setParam(ode.ParamVel, 0.4)  # 
            hinge_joint5.setParam(ode.ParamFMax, 400)  # max力
            hinge_joint6.setParam(ode.ParamVel, -0.4)  # 
            hinge_joint6.setParam(ode.ParamFMax, 400)  # max力
            
            #指の接触センサー
            if grip_counter > 1:
                #指が押し返されたかどうかを判定
                if hinge_joint5.getAngleRate() < 0 or hinge_joint6.getAngleRate() > -0:
                    #指が押し返されたら片方の指を固定
                    hinge_joint6.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
                    hinge_joint6.setParam(ode.ParamFMax, 1000)  # max力   
            grip_counter += 1

        #握った手先を固定
        if grip_counter == 200:        
            #握った手先を固定
            hinge_joint5.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
            hinge_joint5.setParam(ode.ParamFMax, 1000)  # max力
            hinge_joint6.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
            hinge_joint6.setParam(ode.ParamFMax, 1000)  # max力   

        #上昇
        if y <= 0.5 and grip_counter == 200:
            #上昇
            hinge_joint1.setParam(ode.ParamVel, 0.3)  # 
            hinge_joint1.setParam(ode.ParamFMax, 200)  # max力       
            hinge_joint2.setParam(ode.ParamVel, 0.0)  # 
            hinge_joint2.setParam(ode.ParamFMax, 200)  # max力
            hinge_joint3.setParam(ode.ParamVel, -0.3)  # 
            hinge_joint3.setParam(ode.ParamFMax, 200)  # max力  

       #アームの上昇終了判定フラグを立てる
        if y > 0.5 and grip_counter == 200:
            rise_flag = 1
        #上昇終了。まっすぐ戻る。
        if rise_flag == 1 and grip_counter == 200:    
            hinge_joint1.setParam(ode.ParamVel, 0.3)  # 速度
            hinge_joint1.setParam(ode.ParamFMax, 200)  # max力       
            hinge_joint2.setParam(ode.ParamVel, -0.6)  # 速度
            hinge_joint2.setParam(ode.ParamFMax, 200)  # max力 
            hinge_joint3.setParam(ode.ParamVel, 0.3)  # 速度
            hinge_joint3.setParam(ode.ParamFMax, 200)  # max力  

        #アームの腕を上限まで曲げた状態で、根本の回転をストップ。
        if hinge_joint1.getAngle() > 1.45 and grip_counter == 200:
            hinge_joint1.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
            hinge_joint1.setParam(ode.ParamFMax, 1000)  # max力 
            hinge_joint3.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
            hinge_joint3.setParam(ode.ParamFMax, 1000)  # max力             
       
        #アームの腕を上限まで曲げた状態で、肘の回転をストップ。
        if hinge_joint2.getAngle() < -2.90 and grip_counter == 200:
            hinge_joint2.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
            hinge_joint2.setParam(ode.ParamFMax, 1000)  # max力 

            #ボールを運べたかどうかのクラスラベルを取得
            if global_pick_up_balls_flag == 2:
                global_pick_up_balls_flag = 0   #ボールを拾うタイミングと離すタイミングのフラグを0にする
                #ボールを運べたかどうかのフラグを立てる
                pick_up_success_flag = 0
                for v in bodies:
                    ball_x, ball_y, ball_z = v.getPosition()
                    #print(ball_x)
                    if ball_x > -0.6 and ball_y > 0.1:
                        pick_up_success_flag = 1

                #ボールをアームの根元まで運べたとき
                if pick_up_success_flag == 1:
                    class_label.append(1)
                #ボールをアームの根元まで運べなかったとき
                else:
                    class_label.append(0)                
                #print(class_label)
            
            #手先の指を離す
            hinge_joint5.setParam(ode.ParamVel, -1.0)  # 
            hinge_joint5.setParam(ode.ParamFMax, 200)  # max力
            hinge_joint6.setParam(ode.ParamVel, 1.0)  # 
            hinge_joint6.setParam(ode.ParamFMax, 200)  # max力
            #手先の指をストップ
            if hinge_joint5.getAngle() < -0.02:
                hinge_joint5.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
                hinge_joint5.setParam(ode.ParamFMax, 200)  # max力    
            if  0.02 < hinge_joint6.getAngle():
                hinge_joint6.setParam(ode.ParamVel, 0.0)  # 速度をゼロに設定
                hinge_joint6.setParam(ode.ParamFMax, 200)  # max力   
            #手先の指がストップしているとき
            if hinge_joint5.getAngle() < -0.02 and hinge_joint6.getAngle() > 0.02:     
                #カウンターを最初に戻す
                counter = 0
                
                #drop_sphere_positionが2.0以上になったら、ラベルデータを保存して、カウンターのループを抜ける。
                if drop_sphere_position < -2.0:               
                    counter = 701
                    with open("label_data_for_learning_ResNet18.txt","w") as o:
                        for index, v in enumerate(class_label):
                            if index == len(class_label) - 1:
                                print(str(v), end="", file=o)
                            else:
                                print(str(v) + ",", end="", file=o)

    #異なる視点の画像を2つの画面に描画する
    glutSetWindow(subwinnum[0])
    glutDisplayFunc (_drawfunc0)
    glutPostRedisplay ()

    glutSetWindow(subwinnum[1])
    glutDisplayFunc (_drawfunc1)
    glutPostRedisplay ()

     ##衝突検出部分を書き換え。#############
    # Simulate
    n = 4
    for i in range(n):
        for g1 in geoms_robo:    
            for g2 in geoms:
                near_callback((world,contactgroup), g1, g2) #ボールとアームロボットの衝突
 
        for g1 in geoms_robo:
            near_callback((world,contactgroup), g1, floor)  #床と箱ロボットの衝突

        for g1 in geoms:
            near_callback((world,contactgroup), g1, floor) #床とボールの衝突
            for g2 in geoms:
                near_callback((world,contactgroup), g1, g2) #ボール同士の衝突     

        #space.collide((world,contactgroup), ode.collide_callback(g1, floor))
        # Simulation step
        world.step(dt/n)
        # Remove all contact joints
        contactgroup.empty()

     ##衝突検出部分を書き換え。終了。#############

    lasttime = time.time()


glutIdleFunc (_idlefunc)

glutMainLoop ()

