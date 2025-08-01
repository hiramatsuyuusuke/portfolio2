#ODE0.16.4付属のtutorial3.pyを変更して作成したソースコードです
#3D空間で学習用画像と学習用クラスラベルを作成します

import sys, time
from math import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import ode

from PIL import Image
from PIL import ImageOps

import numpy as np


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

    body, geom = create_cylinder(world, space, density, 3, r, h)  #odeとopenglのシリンダーの方向を一致させるために、3(z軸方向)にする。
    
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

# drop_box_robo_object
def drop_box_robo( lx, ly, lz, px, py, pz, density):
    """Drop an object into the scene."""
    global bodies_robo, geoms_robo, objcount

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
        c.setBounce(0)
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
    text = "test"
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))  # 文字を描画   

    glRasterPos2f(-0.95, -0.89)  # 描画位置を指定
    text = "test"
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))  # 文字を描画   


    glRasterPos2f(0.05, -0.6)  # 描画位置を指定
    text = "The view from the green box robot."
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))  # 文字を描画   

    glRasterPos2f(0.05, -0.78)  # 描画位置を指定
    text = "test"
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))  # 文字を描画   

    glRasterPos2f(0.05, -0.89)  # 描画位置を指定
    text = "test"
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

#テクスチャポリゴン
def draw_tex_polygon(tex_floor, tex_wall):

    glMaterialfv(GL_FRONT, GL_AMBIENT, [0.5, 0.5, 0.5, 0.5])  #環境光の影響  
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])#地の色の設定

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)#テクスチャの色と地の色の混ざり方の設定
    glEnable(GL_BLEND)
    glEnable(GL_TEXTURE_2D)
    glNormal3f(0, 1, 0)     #glNormal3f()は非推奨の関数らしい

    #床
    glBindTexture(GL_TEXTURE_2D, tex_floor, )
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
#俯瞰図用のサブウィンドウの作成
glutInitWindowSize ( 320, 320);
subwinnum.append(glutCreateSubWindow(winnum, 10, 10, 320, 320))
glutDisplayFunc(draw2)

#箱ロボットの視界用のサブウィンドウの作成
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

#counterイベント時間の設定
timing_update_position_and_clear_box = 100
timing_create_box = 102
timing_judge1 = 107         #無効座標の判定。#timing_create_boxとtiming_judge1の間は、3カウント以上開ける。
timing_img_capture = 109    #箱ロボットの視界をキャプチャ。
timing_rolling_box = 111
timing_judge2 = 131         #分類ラベルを判定。

#箱ロボットと障害物の衝突判定用フラグ
global_robo_obstacle_collision_flag = 0 

#北=1
gaze_x = 0 #視線方向の初期値をセット
gaze_z = -10 #視線方向の初期値をセット
Force_x = 0 #箱を押す力の方向の初期値をセット
Force_z = -120  #箱を押す力の方向の初期値をセット

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

if rolling_direc_count == 0:
    # ゼロ埋め2D配列を作成
    invalid_position_matrix = [[0 for _ in range(ipms)] for _ in range(ipms)]#無効座標ファイル作成用

if rolling_direc_count != 0:
    # Read lines from a file
    with open('invalid_position_matrix.txt', 'r') as file:
        lines = file.readlines()
    # Convert lines to a 2D NumPy array (assuming space-separated values)
    invalid_position_matrix = np.array([list(map(int, line.strip().split(","))) for line in lines])

# 配列の省略表示を無効化
#np.set_printoptions(threshold=np.inf)
#print(invalid_position_matrix)

#学習用のクラスラベル
class_label = []  

#テクスチャ読み込み#
glutSetWindow(subwinnum[0])
tex_floor = load_texture("sample1.png")
tex_wall = load_texture("sample2.png")
glutSetWindow(subwinnum[1])
tex_floor = load_texture("sample1.png")
tex_wall = load_texture("sample2.png")

#障害物の作成
#ベッドの天板
drop_box( 2.2, 0.2, 1.2,    
        1.0, 0.3, -3.5, 1000)  #(lx, ly, lz, px, py, pz, density)   
#ベッドの足#####
drop_cylinder( 2, 0.1, 0.2,     
        0., 0.1, -4., 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
drop_cylinder( 2, 0.1, 0.2,     
        2., 0.1, -4., 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
drop_cylinder( 2, 0.1, 0.2,     
        2., 0.1, -3., 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
drop_cylinder( 2, 0.1, 0.2,     
        0., 0.1, -3., 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。

#扇風機の頭1
drop_cylinder( 1, 0.3, 0.1,     
        0, 0.7, -0.25 + 2.0, 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
#扇風機の頭2
drop_cylinder( 1, 0.1, 0.4,     
        0, 0.7, 0. + 2.0, 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
#扇風機の足1
drop_cylinder( 2, 0.4, 0.1,     
        0, 0.05, 0. + 2.0, 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
#扇風機の足2
drop_cylinder( 2, 0.1, 0.6,     
        0, 0.3, 0. + 2.0, 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。

#机の天板
drop_box( 1.3, 0.2, 1.7,    
        3.8, 0.7, 0.825, 1000)  #(lx, ly, lz, px, py, pz, density)
#机の脚
drop_box( 0.15, 0.6, 0.15, 3.3, 0.3, 0., 1000)  #(lx, ly, lz, px, py, pz, density)  
drop_box( 0.15, 0.6, 0.15, 4.3, 0.3, 0., 1000)  #(lx, ly, lz, px, py, pz, density)  
#机の引き出し
drop_box( 1.0, 0.6, 0.5,    
        3.8, 0.3, 1.4, 1000)  #(lx, ly, lz, px, py, pz, density)     

#棚の天板1
drop_box( 1.0, 0.1, 2.0, 0.5 - 4.2, 0.6, 1.0, 1000)  #(lx, ly, lz, px, py, pz, density)  
#棚の天板2
drop_box( 1.0, 0.1, 2.0, 0.5 - 4.2, 1.0, 1.0, 1000)  #(lx, ly, lz, px, py, pz, density)  
#棚の脚
drop_box( 0.15, 1., 0.15, 0. - 4.2, 0.5, 0., 1000)  #(lx, ly, lz, px, py, pz, density)  
drop_box( 0.15, 1., 0.15, 1. - 4.2, 0.5, 0., 1000)  #(lx, ly, lz, px, py, pz, density)  
drop_box( 0.15, 1., 0.15, 1. - 4.2, 0.5, 2., 1000)  #(lx, ly, lz, px, py, pz, density)  
drop_box( 0.15, 1., 0.15, 0. - 4.2, 0.5, 2., 1000)  #(lx, ly, lz, px, py, pz, density)      

#外周
drop_box( 0.3, 1.0, 8.99,
            -4.651, 0.5001, 0.001, 1000)  #(lx, ly, lz, px, py, pz, density)
drop_box( 0.3, 1.0, 8.99,
            4.651, 0.5001, 0.001, 1000)  #(lx, ly, lz, px, py, pz, density)
drop_box( 8.99, 1.0, 0.3,
            0.001, 0.5001, 4.651, 1000)  #(lx, ly, lz, px, py, pz, density)
drop_box( 8.99, 1.0, 0.3,
            0.001, 0.5001, -4.651, 1000)  #(lx, ly, lz, px, py, pz, density)

# 障害物用の固定ジョイントの作成
fixed_joints=[]
for i in range(19):
    fixed_joints.append(ode.FixedJoint(world))
    fixed_joints[i].attach(bodies[i], None)  # ボディを固定
    fixed_joints[i].setFixed()

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
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.5, 0.5, 0.5, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.5, 0.5, 1.0])
        #扇風機のindex
        if 5 <= index and index <= 8:            
            glMaterialfv(GL_FRONT, GL_AMBIENT, [1.0, 1.0, 1.0, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        #机のindex
        if 9 <= index and index <= 12:            
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.4, 0.23, 0.2, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.4, 0.23, 0.2, 1.0])
        #棚のindex
        if 13 <= index and index <= 18:            
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.2, 0.2, 0.2, 1.0])
        draw_body(b)

    for index, b in enumerate(bodies_robo):
        if index == 0:             
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.5, 1, 0.5, 0.5])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 1, 0.5, 0.1])
        draw_body(b)


    draw_tex_polygon(tex_floor, tex_wall)

    glutSwapBuffers ()

def _drawfunc1 ():
    global bodies, gaze_x, gaze_z
    global bodies_robo
    # Draw the scene
    prepare_GL()
    
    x,y,z = bodies_robo[0].getPosition()#箱ロボットの座標

    #箱ロボットの視点
    gluLookAt ( x, y, z, x + gaze_x, 0.1, z + gaze_z, 0, 1, 0)#（視点位置、注視点位置、姿勢方向）

    for index, b in enumerate(bodies):
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.3, 0.3, 1, 1.0])  #環境光の影響  
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.3, 0.3, 1, 1.0])     
        #ベッドのindex
        if 0 <= index and index <= 4:            
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.5, 0.5, 0.5, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.5, 0.5, 1.0])
        #扇風機のindex
        if 5 <= index and index <= 8:            
            glMaterialfv(GL_FRONT, GL_AMBIENT, [1.0, 1.0, 1.0, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        #机のindex
        if 9 <= index and index <= 12:            
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.4, 0.23, 0.2, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.4, 0.23, 0.2, 1.0])
        #棚のindex
        if 13 <= index and index <= 18:            
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.2, 0.2, 0.2, 1.0])

        draw_body(b)

    draw_tex_polygon(tex_floor, tex_wall)

    glutSwapBuffers ()

#glutDisplayFunc (_drawfunc)

# idle callback
def _idlefunc ():
    global world, contactgroup
    global counter, lasttime
    global bodies, geoms, bodies_robo, geoms_robo
    global gaze_x, gaze_z, Force_x, Force_z
    global rolling_direc_count, rolling_direc_count_max
    global box_px, box_pz, box_dpx, box_dpz, box_px_start, box_pz_start, box_px_end, box_pz_end
    global class_label, invalid_position_matrix, ipmx, ipmz
    global timing_update_position_and_clear_box, timing_create_box, timing_judge1, timing_img_capture, timing_rolling_box, timing_judge2
    global global_robo_obstacle_collision_flag
    global subwinnum
    #t = dt - (time.time() - lasttime)
    #if (t > 0):
        #time.sleep(t)
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
                        gaze_x = 10 
                        gaze_z = 0 
                        Force_x = 120 
                        Force_z = 0 
                    #s
                    if rolling_direc_count == 3:
                        gaze_x = 0                
                        gaze_z = 10
                        Force_x = 0
                        Force_z = 120
                    #w
                    if rolling_direc_count == 4:
                        gaze_x = -10 
                        gaze_z = 0
                        Force_x = -120 
                        Force_z = 0

                    counter = timing_update_position_and_clear_box - 1

                box_px = box_px_start     #箱のx方向座標を折り返し
                ipmx = 0    #無効座標ファイル作成用
            #箱ロボットをクリア
            bodies_robo.clear()
            geoms_robo.clear()

        #箱ロボットを作成     
        if counter == timing_create_box:
            #無効座標を取得するとき
            if rolling_direc_count == 0:
                #無効座標ファイル作成用に一回り大きい箱ロボットを作成する
                drop_box_robo(0.4, 0.4, 0.4,
                            box_px, 0.21, box_pz, 1000.0)  #(lx, ly, lz, px, py, pz, density)
            #衝突判定をするとき
            else:
                #箱ロボットを作成する。一回り大きい箱ロボットを作成する
                drop_box_robo(0.3, 0.3, 0.3,
                            box_px, 0.15, box_pz, 10.0)  #(lx, ly, lz, px, py, pz, density)    
                    
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

        #箱の作成が終わった後から描画開始する。視点座標に箱の座標を使っているため。
        if counter > timing_create_box + 1:        
            #異なる視点の画像を2つの画面に描画する
            glutSetWindow(subwinnum[0])
            glutDisplayFunc (_drawfunc0)
            glutPostRedisplay ()

            glutSetWindow(subwinnum[1])
            glutDisplayFunc (_drawfunc1)
            glutPostRedisplay ()

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
                near_callback((world,contactgroup), g1, floor)#障害物と床のgeoms
            for g1 in geoms_robo:
                near_callback((world,contactgroup), g1, floor)#ロボと床のgeoms

            #space.collide((world,contactgroup), ode.collide_callback(g1, floor))
            # Simulation step
            world.step(dt/n)
            # Remove all contact joints
            contactgroup.empty()
        ##衝突検出部分を書き換え。終了。#############
        
    lasttime = time.time()

glutIdleFunc (_idlefunc)

glutMainLoop ()

