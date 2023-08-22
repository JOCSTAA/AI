import turtle as t


class Paddle():

    def __init__(self):

        self.done = False
        self.reward = 0
        self.hit, self.miss = 0, 0
        self.startx = 0
        self.starty = 0
        self.width = 256
        self.height = 256
        self.paddleleftboundary = self.width - 175
        self.paddlerightboundary = -self.width + 175
        self.paddlecenter = -100

        # Setup Background

        self.win = t.Screen()
        self.win.title('Paddle')
        self.win.bgcolor('black')
        self.win.setup(width=self.width, height=self.height, startx=self.startx, starty=self.starty)
        # self.win.screensize(self.width, self.height)
        # self.win.setup(width=1, height=1, startx=None, starty=None)
        self.win.tracer(0)

        # Paddle

        self.paddle = t.Turtle()
        self.paddle.speed(0)
        self.paddle.shape('square')
        self.paddle.shapesize(stretch_wid=0.5, stretch_len=2.5)
        self.paddle.color('white')
        self.paddle.penup()
        self.paddle.goto(0, self.paddlecenter)

        # Ball

        self.ball = t.Turtle()
        self.ball.speed(0)
        self.ball.shape('circle')
        self.ball.color('red')
        self.ball.penup()
        self.ball.goto(0, 100)
        self.ball.dx = 3
        self.ball.dy = -3

        # Score

        self.score = t.Turtle()
        self.score.speed(0)
        self.score.color('white')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, 250)
        self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center',
                         font=('Courier', 24, 'normal'))

        # -------------------- Keyboard control ----------------------

        self.win.listen()
        self.win.onkey(self.paddle_right, 'Right')
        self.win.onkey(self.paddle_left, 'Left')

    # Paddle movement

    def paddle_right(self):

        x = self.paddle.xcor()
        if x < self.paddleleftboundary:
            self.paddle.setx(x + 20)

    def paddle_left(self):

        x = self.paddle.xcor()
        if x > self.paddlerightboundary:
            self.paddle.setx(x - 20)

    def run_frame(self):

        self.win.update()

        # Ball moving

        self.ball.setx(self.ball.xcor() + self.ball.dx)
        self.ball.sety(self.ball.ycor() + self.ball.dy)

        # Ball and Wall collision

        if self.ball.xcor() > self.paddleleftboundary:
            self.ball.setx(self.paddleleftboundary)
            self.ball.dx *= -1

        if self.ball.xcor() < self.paddlerightboundary:
            self.ball.setx(self.paddlerightboundary)
            self.ball.dx *= -1

        if self.ball.ycor() > self.paddleleftboundary:
            self.ball.sety(self.paddleleftboundary)
            self.ball.dy *= -1

        # Ball Ground contact

        if self.ball.ycor() < self.paddlecenter:
            self.ball.goto(0, 100)
            self.miss += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center',
                             font=('Courier', 24, 'normal'))
            self.reward -= 3
            self.done = True

        # Ball Paddle collision

        if abs(self.ball.ycor() + 100) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 55:
            self.ball.dy *= -1
            self.hit += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center',
                             font=('Courier', 24, 'normal'))
            self.reward += 3

    # ------------------------ AI control ------------------------

    # 0 move left
    # 1 do nothing
    # 2 move right

    def reset(self):

        self.paddle.goto(0, -275)
        self.ball.goto(0, 100)
        return [self.paddle.xcor() * 0.01, self.ball.xcor() * 0.01, self.ball.ycor() * 0.01, self.ball.dx, self.ball.dy]

    def step(self, action):

        self.reward = 0
        self.done = 0

        if action == 0:
            self.paddle_left()
            self.reward -= .1

        if action == 2:
            self.paddle_right()
            self.reward -= .1

        self.run_frame()

        state = [self.paddle.xcor() * 0.01, self.ball.xcor() * 0.01, self.ball.ycor() * 0.01, self.ball.dx,
                 self.ball.dy]
        return self.reward, state, self.done

# ------------------------ Human control ------------------------

# env = Paddle()
#
# while True:
#      env.run_frame()
