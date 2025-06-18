#ODE0.16.4付属のtutorial3.pyを変更して作成したソースコードです
#3D空間で箱と障害物の衝突を検証して、箱視点の学習用画像と学習用クラスラベルを作成します

import sys, time
from math import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import ode

from PIL import Image
from PIL import ImageOps

import copy

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
def drop_object( lx, ly, lz, px, py, pz, density):
    """Drop an object into the scene."""

    global bodies, geoms, objcount

    body, geom = create_box(world, space, density, lx, ly, lz)
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

#テクスチャポリゴン
def draw_tex_polygon():

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

#n
gaze_x = 0 #視線方向の初期値をセット
gaze_z = -10 #視線方向の初期値をセット
Force_x = 0 #箱を押す力の方向の初期値をセット
Force_z = -100  #箱を押す力の方向の初期値をセット
"""
#e
gaze_x = 10 
gaze_z = 0 
Force_x = 100 
Force_z = 0 

#s
gaze_x = 0                
gaze_z = 10
Force_x = 0
Force_z = 100

#w
gaze_x = -10 
gaze_z = 0
Force_x = -100 
Force_z = 0
"""

capimgnum = 0
rolling_direc_count_max = 4

rolling_direc_count = 0   #箱を転がす方角のカウント。北=1、東=2、南=3、西=4
ops0=[]  #箱が作成される前の障害物の座標。
ops1=[]  #箱が転がる前の障害物の座標。箱と障害物が重なっていれば、障害物が移動する。
ops2=[]  #箱が転がった後の障害物の座標。箱と障害物が衝突すれば、障害物が移動する。
class_label = []  #学習用のクラスラベル

box_px_start = -4.0
box_pz_start = -4.0
box_px_end = 4.0
box_pz_end = 4.0
box_px = box_px_start  #箱のx座標の初期値をセット
box_pz = box_pz_start  #箱のy座標の初期値をセット
box_dpx = 0.25  #箱のx座標の探索の刻み幅
box_dpz = 0.25  #箱のy座標の探索の刻み幅


#テクスチャ読み込み#
glutSetWindow(subwinnum[0])
tex_floor = load_texture("sample1.png")
tex_wall = load_texture("sample2.png")
glutSetWindow(subwinnum[1])
tex_floor = load_texture("sample1.png")
tex_wall = load_texture("sample2.png")


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
        if index == len(bodies)-1:
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.5, 1, 0.5, 0.5])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 1, 0.5, 0.1])

        if index == 0:                 
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
        x,y,z = bodies[len(bodies)-1].getPosition()

    #箱ロボットの視点
    gluLookAt ( x, y, z, x + gaze_x, 0.1, z + gaze_z, 0, 1, 0)#（視点位置、注視点位置、姿勢方向）

    for index, b in enumerate(bodies):
        if index == 0:                 
            glMaterialfv(GL_FRONT, GL_AMBIENT, [0.3, 0.3, 1, 1.0])  #環境光の影響  
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.3, 0.3, 1, 1.0])

        if index < len(bodies)-1:           
            draw_body(b)

    draw_tex_polygon()

    glutSwapBuffers ()

#glutDisplayFunc (_drawfunc)

# idle callback
def _idlefunc ():
    global counter, lasttime
    global gaze_x, gaze_z, Force_x, Force_z
    global bodies, geoms, subwinnum, world,contactgroup
    global rolling_direc_count, ops0, ops1, ops2, class_label
    global box_px, box_pz, box_dpx, box_dpz, box_px_start, box_pz_start, box_px_end, box_pz_end

    #t = dt - (time.time() - lasttime)
    #if (t > 0):
        #time.sleep(t)

    if rolling_direc_count <= rolling_direc_count_max:#探索する方角の回数が4以下のとき。北東南西の４回。
        counter += 1
        if counter==181:  

            box_px += box_dpx  #箱のx座標を更新

            bodies.clear()
            geoms.clear()
            ops1.clear()  #箱が転がる前の障害物の座標。箱と障害物が重なっていれば、障害物が移動する。
            ops2.clear()  #箱が転がった後の障害物の座標。箱と障害物が衝突すれば、障害物が移動する。

            #room1
            drop_object(0.3, 1.0, 6.0,
                        2.0, 0.51, 1.49, 1e-6)  #(lx, ly, lz, px, py, pz, density)       
            drop_object(0.3, 1.0, 6.0,
                        -2.0, 0.51, -1.49, 1e-6)  #(lx, ly, lz, px, py, pz, density)   
        
            #外周
            drop_object(0.3, 1.0, 8.99,
                        -4.651, 0.51, 0.001, 1e-6)  #(lx, ly, lz, px, py, pz, density)  
            drop_object(0.3, 1.0, 8.99,
                        4.651, 0.51, 0.001, 1e-6)  #(lx, ly, lz, px, py, pz, density)             
            drop_object(8.99, 1.0, 0.3,
                        0.001, 0.51, 4.651, 1e-6)  #(lx, ly, lz, px, py, pz, density)        
            drop_object(8.99, 1.0, 0.3,
                        0.001, 0.51, -4.651, 1e-6)  #(lx, ly, lz, px, py, pz, density)  
         
            """        
            #room2
            drop_object(2.0, 1.0, 0.3,
                        0.0, 3.0, -2.0, 1e-5)  #(lx, ly, lz, px, py, pz, density)             
            drop_object(0.3, 1.0, 2.0,
                        1.2, 3.0, -1.15, 1e-5)  #(lx, ly, lz, px, py, pz, density)                
            drop_object(0.3, 1.0, 2.0,
                        -1.2, 3.0, -1.15, 1e-5)  #(lx, ly, lz, px, py, pz, density) 
            
            drop_object(2.0, 1.0, 0.3,
                        0.0, 3.0, 2.5, 1e-5)  #(lx, ly, lz, px, py, pz, density)  
            drop_object(0.3, 1.0, 2.0,
                        2.0, 3.0, 1.55, 1e-5)  #(lx, ly, lz, px, py, pz, density)  
            drop_object(0.3, 1.0, 2.0,
                        -2.0, 3.0, 1.55, 1e-5)  #(lx, ly, lz, px, py, pz, density)  

            drop_object(0.3, 1.0, 1.0,
                        0.0, 3.0, 4.0, 1e-5)  #(lx, ly, lz, px, py, pz, density)     
            drop_object(0.3, 1.0, 1.0,
                        1.2, 3.0, -4.0, 1e-5)  #(lx, ly, lz, px, py, pz, density)             
            """  
 
                    
        #転がる前の箱と障害物が重なっていないかを判定。オブジェクトを落下させているので、counterを少し遅らせる。
        if counter==190:     

            #始めの一回だけ、箱ロボットと障害物が重なっていない状態の座標を保持。探索の最初に箱と障害物が重なっていないことが必要。
            if rolling_direc_count == 0:
                rolling_direc_count = 1
                #rolling_direc_count = rolling_direc_count_max
                ##始めの一回だけ、箱が作成される前の障害物の座標を取得
                for index, b in enumerate(bodies):
                    if index < len(bodies):
                        ops0.append(b.getPosition ())
            #西=4
            if rolling_direc_count == 4:
                #箱ロボットを最後に作成する
                drop_object(0.3, 0.3, 0.3,
                            -box_px, 0.16, box_pz, 10.0)  #(lx, ly, lz, px, py, pz, density)                           
            #東=2
            if rolling_direc_count == 2:
                #箱ロボットを最後に作成する
                drop_object(0.3, 0.3, 0.3,
                            box_px, 0.16, box_pz, 10.0)  #(lx, ly, lz, px, py, pz, density)
            #北=1
            if rolling_direc_count == 1:
                #箱ロボットを最後に作成する
                drop_object(0.3, 0.3, 0.3,
                            box_pz, 0.16, -box_px, 10.0)  #(lx, ly, lz, px, py, pz, density)   
            #南=3              
            if rolling_direc_count == 3:
                #箱ロボットを最後に作成する
                drop_object(0.3, 0.3, 0.3,
                            box_pz, 0.16, box_px, 10.0)  #(lx, ly, lz, px, py, pz, density)   

        if counter==200:    
            #障害物の座標を取得
            for index, b in enumerate(bodies):
                if index < len(bodies)-1:
                    ops1.append(b.getPosition ())
            #最初に箱と障害物が重なっているとき（障害物が移動するので、座標が変化している）
            if ops0 != ops1:
                print(2)
                counter = 180   #カウンターを最初に戻す。画像のキャプチャとラベルの取得は行わない。

        #学習用の画像のキャプチャ
        if counter==201:
            capture2().save( "img/test" + str(capimgnum + len(class_label)) + ".jpg")  # 縮小した画像を保存

        #箱を転がす 
        if counter==202:     
            bodies[len(bodies)-1].addForce(( Force_x, 0, Force_z))
        if counter==210:     
            bodies[len(bodies)-1].addForce(( Force_x, 0, Force_z))       

        #衝突するか衝突しないかの学習用クラスラベルデータの取得
        if counter==240:   
            #現在の障害物の座標を取得
            for index, b in enumerate(bodies):
                if index < len(bodies)-1:
                    ops2.append(b.getPosition ())

            #衝突するか衝突しないかの学習用クラスラベルを取得          
            if ops0 == ops2:
                print(0)
                class_label.append(0)
            else:
                print(1)
                class_label.append(1)

            counter = 180   #カウンターを最初に戻す
        
            #4つの方角（北東南西）の処理が終わったときに衝突判定の学習用クラスラベルデータを出力
            if box_px > box_px_end and box_pz > box_pz_end and rolling_direc_count == rolling_direc_count_max:
                with open("box_collision_class_label.txt","w") as o:
                    for index, v in enumerate(class_label):
                        if index == len(class_label) - 1:
                            print(str(v), end="", file=o)
                        else:
                            print(str(v) + ",", end="", file=o)

            #一つの方角の探索が終わったとき
            if box_px > box_px_end and box_pz > box_pz_end:
                rolling_direc_count += 1 #rolling_direc_countを次の方角に更新

                #次の方角の探索の始まりの座標
                box_px = box_px_start
                box_pz = box_pz_start
                #e
                if rolling_direc_count == 2:
                    gaze_x = 10 
                    gaze_z = 0 
                    Force_x = 100 
                    Force_z = 0 
                #s
                if rolling_direc_count == 3:
                    gaze_x = 0                
                    gaze_z = 10
                    Force_x = 0
                    Force_z = 100
                #w
                if rolling_direc_count == 4:
                    gaze_x = -10 
                    gaze_z = 0
                    Force_x = -100 
                    Force_z = 0

            #箱のx方向座標がmaxを超えたとき
            if box_px > box_px_end:
                box_px = box_px_start     #箱のx方向座標を折り返し
                box_pz += box_dpz       #箱のy方向座標を更新

        #箱の作成が終わった後で描画するようにする。視点座標に箱の座標を使っているため。
        if counter > 190:  
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

