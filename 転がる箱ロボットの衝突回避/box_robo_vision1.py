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

# drop_object
def drop_object( lx, ly, lz, px, py, pz):
    """Drop an object into the scene."""

    global bodies, geoms, objcount

    body, geom = create_box(world, space, 10, lx, ly, lz)
    body.setPosition( (px, py, pz) )
    theta = 0
    ct = cos (theta)
    st = sin (theta)
    body.setRotation([ct, 0., -st, 0., 1., 0., st, 0., ct])
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
    """
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
    """
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

# 箱ロボットの行動を決める値
judge = 0
judge_pattern = 2
gaze_x = 0
gaze_z = -10
Force_x = 0
Force_z = -120

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


# keyboard callback
def _keyfunc (c, x, y):
    sys.exit (0)

glutKeyboardFunc (_keyfunc)

# draw callback
def _drawfunc0 ():
    global bodies
    # Draw the scene
    prepare_GL()

    #俯瞰の視点
    gluLookAt (3.0*2.0, 3.6*2.0, 5.0*1.5, -1.0, -1.0, 0, 0, 1, 0)#（視点位置、注視点位置、姿勢方向）

    for index, b in enumerate(bodies):
        if index == 0:
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.5, 1, 0.5, 0.5])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 1, 0.5, 0.1])
        if index == 1:                 
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.3, 0.3, 1, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.3, 0.3, 1, 1.0])
        draw_body(b)

    draw_tex_polygon()

    glutSwapBuffers ()

def _drawfunc1 ():
    global bodies, objcount, gaze_x, gaze_z
    # Draw the scene
    prepare_GL()
    
    x,y,z = 0,0,0
    if objcount >= 1:
        x,y,z = bodies[0].getPosition()
    #箱ロボットの視点
    gluLookAt ( x, y, z, x + gaze_x, 0.2, z + gaze_z, 0, 1, 0)#（視点位置、注視点位置、姿勢方向）


    for index, b in enumerate(bodies):
        if index == 1:                 
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.3, 0.3, 1, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.3, 0.3, 1, 1.0])

        if index >= 1:           
            draw_body(b)

    draw_tex_polygon()

    glutSwapBuffers ()

#glutDisplayFunc (_drawfunc)



# idle callback
def _idlefunc ():
    global counter, lasttime
    global judge, judge_pattern, gaze_x, gaze_z, Force_x, Force_z
    global bodies, geoms, subwinnum, world,contactgroup

    t = dt - (time.time() - lasttime)
    if (t > 0):
        time.sleep(t)

    counter += 1
    if counter==30:       
        drop_object(0.3, 0.3, 0.3,
                     0.0, 3.0, 0.0)  #(lx, ly, lz, px, py, pz)
        
        """       
        drop_object(0.3, 1.0, 6.0,
                     2.0, 3.0, 1.49)  #(lx, ly, lz, px, py, pz)       
        drop_object(0.3, 1.0, 6.0,
                     -2.0, 3.0, -1.49)  #(lx, ly, lz, px, py, pz)   
        """        
        
        drop_object(2.0, 1.0, 0.3,
                     0.0, 3.0, -2.0)  #(lx, ly, lz, px, py, pz)             
        drop_object(0.3, 1.0, 2.0,
                     1.2, 3.0, -1.15)  #(lx, ly, lz, px, py, pz)                
        drop_object(0.3, 1.0, 2.0,
                     -1.2, 3.0, -1.15)  #(lx, ly, lz, px, py, pz) 
        
        drop_object(2.0, 1.0, 0.3,
                     0.0, 3.0, 2.5)  #(lx, ly, lz, px, py, pz)  
        drop_object(0.3, 1.0, 2.0,
                     2.0, 3.0, 1.55)  #(lx, ly, lz, px, py, pz)  
        drop_object(0.3, 1.0, 2.0,
                     -2.0, 3.0, 1.55)  #(lx, ly, lz, px, py, pz)  

        drop_object(0.3, 1.0, 1.0,
                     0.0, 3.0, 4.0)  #(lx, ly, lz, px, py, pz)     
        drop_object(0.3, 1.0, 1.0,
                     1.2, 3.0, -4.0)  #(lx, ly, lz, px, py, pz)             


    if counter==200:     
        judge = display_image_recognition()
        if judge == 0:
            bodies[0].addForce(( Force_x, 0, Force_z))

        if judge == 1:
            judge_pattern += 1
            #judge_pattern = random.randint(1, 4)

            if judge_pattern== 1:
                Force_x = 120
                gaze_x = 10
                Force_z = 0
                gaze_z = 0
            if judge_pattern== 2:
                Force_x = 0
                gaze_x = 0
                Force_z = -120
                gaze_z = -10
            if judge_pattern== 3:
                Force_x = -120
                gaze_x = -10
                Force_z = 0
                gaze_z = 0                
            if judge_pattern== 4:
                Force_x = 0
                gaze_x = 0
                Force_z = 120
                gaze_z = 10

            if judge_pattern == 4:
                judge_pattern = 0

        #カウンターをオブジェクト作成後に戻す
        counter = 160
  
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





glutIdleFunc (_idlefunc)

glutMainLoop ()

