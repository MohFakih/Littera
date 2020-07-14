import pygame as pg


class PGWindow:
    def __init__(self, size=(1024, 512), fps=30):
        # PyGame formalities
        pg.init()
        self.size = size
        self.fps = fps
        self.screen = pg.display.set_mode(self.size)
        pg.display.set_caption("Littera")

        # Needed flags
        self.exitFlag = False
        self.clock = pg.time.Clock()  # Needed to limit framerate
        print(type(self.screen))

    def handleEvent(self, event: pg.event):
        pass

    def handleInputs(self):
        pass

    def draw(self):
        pass

    def eventHandleLoop(self):
        # Handle events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.exitFlag = True
            else:
                self.handleEvent(event)

    def runMainLoop(self):
        while True:
            # Limit Framerate
            self.clock.tick(self.fps)
            self.eventHandleLoop()
            if self.exitFlag:
                break
            self.handleInputs()
            self.draw()
            pg.display.flip()
