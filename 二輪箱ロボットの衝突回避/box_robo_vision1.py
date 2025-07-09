#ODE-0.16.4のtutorial3.pyを書き換えたコードです。
#https://hiramatsuyuusuke.github.io/portfolio2/product1.html

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

    if body.shape=="cylinder":
        #ODEのcylinder（重心座標）とOpenGLのcylinder（ボトム座標）の座標設定の違いを修整して描画する。

        #シリンダーの外周を作成
        r,h = body.cylindersize
        glTranslated(0.0, 0.0, -h*0.5) #z軸方向に -h*0.5 平行移動
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluCylinder(quadric, r, r, h, 10, 10)  # Base radius, top radius, height, slices, stacks    
        gluDeleteQuadric(quadric)
        
        #シリンダーの上下2枚のディスクを作成
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluDisk(quadric, 0, r, 10, 10)  # Inner radius, outer radius, slices, loops
        
        glTranslated(0.0, 0.0, h) #z軸方向に h 平行移動    
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
def drop_box_robo( lx, ly, lz, px, py, pz):
    """Drop an object into the scene."""

    global bodies_robo, geoms_robo, objcount

    body, geom = create_box(world, space, 1, lx, ly, lz)
    body.setPosition( (px, py, pz) )
    theta = 0
    ct = cos (theta)
    st = sin (theta)
    body.setRotation([ct, 0., -st, 0., 1., 0., st, 0., ct])
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

#テクスチャポリゴン
def draw_tex_polygon():
    #glClear(GL_COLOR_BUFFER_BIT)#prepare_GL()の中で実行される

    glMaterialfv(GL_FRONT, GL_AMBIENT, [0.5, 0.5, 0.5, 0.5])  #環境光の影響  
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])#地の色の設定

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

# 箱ロボットの初期位置
box_robo_start_x = 2.0
box_robo_start_z = 2.0

#テクスチャ読み込み#
glutSetWindow(subwinnum[0])
tex_floor = load_texture("sample1.png")
tex_wall = load_texture("sample2.png")
glutSetWindow(subwinnum[1])
tex_floor = load_texture("sample1.png")
tex_wall = load_texture("sample2.png")

#学習していないResNet18 modelを読み込んでインスタンスを生成
Learned_model = models.resnet18(weights = None)
# Modify the final fully connected layer for a custom number of classes
num_classes = 2 #衝突するクラスと衝突しないクラスの2つ
Learned_model.fc = nn.Linear(Learned_model.fc.in_features, num_classes)
#3D空間の画像を使って学習したResNet18の読み込み
Learned_model.load_state_dict(torch.load("Weight1.pth", weights_only=True))
Learned_model.eval()  # 推論モードに切り替え


#障害物の作成
#ベッドの天板
drop_box(2.2, 0.2, 1.2, 1.0, 0.3, -3.5, 1000)  #(lx, ly, lz, px, py, pz, density)   
#ベッドの足#####
drop_cylinder(2, 0.1, 0.2, 0., 0.1, -4., 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
drop_cylinder(2, 0.1, 0.2, 2., 0.1, -4., 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
drop_cylinder(2, 0.1, 0.2, 2., 0.1, -3., 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
drop_cylinder(2, 0.1, 0.2, 0., 0.1, -3., 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。

#扇風機の頭1
drop_cylinder(1, 0.3, 0.1, 0, 0.7, -0.25 + 2.0, 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
#扇風機の頭2
drop_cylinder(1, 0.1, 0.4, 0, 0.7, 0. + 2.0, 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
#扇風機の足1
drop_cylinder(2, 0.4, 0.1, 0, 0.05, 0. + 2.0, 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
#扇風機の足2
drop_cylinder(2, 0.1, 0.6, 0, 0.3, 0. + 2.0, 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。

#机の天板
drop_box(1.3, 0.2, 1.7, 3.8, 0.7, 0.825, 1000)  #(lx, ly, lz, px, py, pz, density)
#机の脚
drop_box(0.15, 0.6, 0.15, 3.3, 0.3, 0., 1000)  #(lx, ly, lz, px, py, pz, density)  
drop_box(0.15, 0.6, 0.15, 4.3, 0.3, 0., 1000)  #(lx, ly, lz, px, py, pz, density)  
#机の引き出し
drop_box(1.0, 0.6, 0.5, 3.8, 0.3, 1.4, 1000)  #(lx, ly, lz, px, py, pz, density)     

#棚の天板1
drop_box(1.0, 0.1, 2.0, -3.7, 0.6, 1.0, 1000)  #(lx, ly, lz, px, py, pz, density)  
#棚の天板2
drop_box(1.0, 0.1, 2.0, -3.7, 1.0, 1.0, 1000)  #(lx, ly, lz, px, py, pz, density)  
#棚の脚
drop_box(0.15, 1., 0.15, -4.2, 0.5, 0., 1000)  #(lx, ly, lz, px, py, pz, density)  
drop_box(0.15, 1., 0.15, -3.2, 0.5, 0., 1000)  #(lx, ly, lz, px, py, pz, density)  
drop_box(0.15, 1., 0.15, -3.2, 0.5, 2., 1000)  #(lx, ly, lz, px, py, pz, density)  
drop_box(0.15, 1., 0.15, -4.2, 0.5, 2., 1000)  #(lx, ly, lz, px, py, pz, density)   

#障害物の固定ジョイントの作成
fixed_joints=[]
for i in range(19):
#for i in range(38):
    fixed_joints.append(ode.FixedJoint(world))
    fixed_joints[i].attach(bodies[i], None)  # ボディを固定
    fixed_joints[i].setFixed()


#二輪箱ロボットの箱の作成
drop_box_robo(0.3, 0.28, 0.2, box_robo_start_x + 0.0, 0.15, box_robo_start_z + 0.0)  #(lx, ly, lz, px, py, pz)
drop_box_robo(0.05, 0.05, 0.05, box_robo_start_x - 0.15, 0.15, box_robo_start_z + 0.0)  #(lx, ly, lz, px, py, pz)        
fixed_joints.append(ode.FixedJoint(world))
fixed_joints[19].attach(bodies_robo[0], bodies_robo[1])  # ボディを固定
fixed_joints[19].setFixed()

#二輪箱ロボットの車輪の作成1
drop_cylinder(1, 0.15, 0.1, box_robo_start_x + 0.0, 0.15, box_robo_start_z + 0.17, 10)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。     
hinge2_joints=[]
hinge2_joints.append(ode.Hinge2Joint(world))
hinge2_joints[0].attach(bodies_robo[0], bodies[19])
hinge2_joints[0].setAnchor((box_robo_start_x + 0.0, 0.15, box_robo_start_z + 0.0))
hinge2_joints[0].setAxis1((0, 0, 1))  # Set the first axis (e.g., wheel rotation)
hinge2_joints[0].setAxis2((0, 1, 0))  # Set the second axis (e.g., suspension/steering)
# 第2軸の回転を固定
hinge2_joints[0].setParam(ode.ParamVel2, 0)  # 速度をゼロに設定
hinge2_joints[0].setParam(ode.ParamFMax2, 1000)  # 強い力で固定

#二輪箱ロボットの車輪の作成2
drop_cylinder(1, 0.15, 0.1, box_robo_start_x + 0.0, 0.15, box_robo_start_z - 0.17, 10)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。  
hinge2_joints.append(ode.Hinge2Joint(world))
hinge2_joints[1].attach(bodies_robo[0], bodies[20])
hinge2_joints[1].setAnchor((box_robo_start_x + 0.0, 0.15, box_robo_start_z + 0.0))
hinge2_joints[1].setAxis1((0, 0, 1))  # Set the first axis (e.g., wheel rotation)
hinge2_joints[1].setAxis2((0, 1, 0))  # Set the second axis (e.g., suspension/steering)
# 第2軸の回転を固定
hinge2_joints[1].setParam(ode.ParamVel2, 0)  # 速度をゼロに設定
hinge2_joints[1].setParam(ode.ParamFMax2, 1000)  # 強い力で固定


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

    for index, b in enumerate(bodies):
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.3, 0.3, 1, 1.0])  #環境光の影響  
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.3, 0.3, 1, 1.0])     
        #ベッドのindex
        if 0 <= index and index <= 4:            
        #if 0 <= index and index <= 9:             
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.5, 0.5, 0.5, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.5, 0.5, 1.0])
        #扇風機のindex
        if 5 <= index and index <= 8:      
        #if 10 <= index and index <= 17: 
            glMaterialfv(GL_FRONT, GL_AMBIENT, [1.0, 1.0, 1.0, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        #机のindex
        if 9 <= index and index <= 12:
        #if 18 <= index and index <= 25:          
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.4, 0.23, 0.2, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.4, 0.23, 0.2, 1.0])
        #棚のindex
        if 13 <= index and index <= 18:
        #if 26 <= index and index <= 37:      
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.2, 0.2, 0.2, 1.0])
        draw_body(b)

    for index, b in enumerate(bodies_robo):
        if index == 0:             
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
    
    x,y,z = 0,0,0
    if objcount >= 1:
        x,y,z = bodies_robo[0].getPosition()#箱ロボットの座標
        x1,y1,z1 = bodies_robo[1].getPosition()

    #箱ロボットの視点
    gluLookAt ( x, y, z, x1, 0.148, z1, 0, 1, 0)#（視点位置、注視点位置、姿勢方向）

    for index, b in enumerate(bodies):
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.3, 0.3, 1, 1.0])  #環境光の影響  
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.3, 0.3, 1, 1.0])     
        #ベッドのindex
        if 0 <= index and index <= 4:
        #if 0 <= index and index <= 9:  
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.5, 0.5, 0.5, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.5, 0.5, 1.0])
        #扇風機のindex
        if 5 <= index and index <= 8:
        #if 10 <= index and index <= 17:         
            glMaterialfv(GL_FRONT, GL_AMBIENT, [1.0, 1.0, 1.0, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        #机のindex
        if 9 <= index and index <= 12:
        #if 18 <= index and index <= 25: 
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.4, 0.23, 0.2, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.4, 0.23, 0.2, 1.0])
        #棚のindex
        if 13 <= index and index <= 18:
        #if 26 <= index and index <= 37:          
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.2, 0.2, 0.2, 1.0])

        draw_body(b)

    draw_tex_polygon()

    glutSwapBuffers ()
#glutDisplayFunc (_drawfunc)



# idle callback
def _idlefunc ():
    global counter, lasttime
    global bodies, geoms, subwinnum, world, contactgroup
    global bodies_robo, geoms_robo

    t = dt - (time.time() - lasttime)
    if (t > 0):
        time.sleep(t)

    counter += 1

    if counter==50:     
        if display_image_recognition() == 0:
            # Configure joint parameters (optional)
            hinge2_joints[0].setParam(ode.ParamVel, -2.5)  # Set desired velocity
            hinge2_joints[0].setParam(ode.ParamFMax, 100)  # Set maximum force

            # Configure joint parameters (optional)
            hinge2_joints[1].setParam(ode.ParamVel, -2.5)  # Set desired velocity
            hinge2_joints[1].setParam(ode.ParamFMax, 100)  # Set maximum force

        if display_image_recognition() == 1:
            # Configure joint parameters (optional)
            hinge2_joints[0].setParam(ode.ParamVel, -2.5)  # Set desired velocity
            hinge2_joints[0].setParam(ode.ParamFMax, 100)  # Set maximum force

            # Configure joint parameters (optional)
            hinge2_joints[1].setParam(ode.ParamVel, 2.5)  # Set desired velocity
            hinge2_joints[1].setParam(ode.ParamFMax, 100)  # Set maximum force

    if counter == 100:  
        # Configure joint parameters (optional)
        hinge2_joints[0].setParam(ode.ParamVel, 0)  # Set desired velocity
        hinge2_joints[0].setParam(ode.ParamFMax, 100)  # Set maximum force

        # Configure joint parameters (optional)
        hinge2_joints[1].setParam(ode.ParamVel, 0)  # Set desired velocity
        hinge2_joints[1].setParam(ode.ParamFMax, 100)  # Set maximum force
        #カウンターを最初に戻す
        counter = 0
  
 
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
                near_callback((world,contactgroup), g1, g2) #障害物と箱ロボットの衝突

        for g1 in geoms_robo:
            near_callback((world,contactgroup), g1, floor)  #床と箱ロボットの衝突
        for g1 in geoms:
            near_callback((world,contactgroup), g1, floor) #床と障害物の衝突

        #space.collide((world,contactgroup), ode.collide_callback(g1, floor))
        # Simulation step
        world.step(dt/n)
        # Remove all contact joints
        contactgroup.empty()

     ##衝突検出部分を書き換え。終了。#############

    lasttime = time.time()


glutIdleFunc (_idlefunc)

glutMainLoop ()

