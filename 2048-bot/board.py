from tkinter import *
from random import randint

SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
STATUS_COLOR = "#ababab"
BACKGROUND_COLOR_DICT = {   2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563", \
                            32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", \
                            512:"#edc850", 1024:"#edc53f", 2048:"#edc22e" }
CELL_COLOR_DICT = { 2:"#776e65", 4:"#776e65", 8:"#f9f6f2", 16:"#f9f6f2", \
                    32:"#f9f6f2", 64:"#f9f6f2", 128:"#f9f6f2", 256:"#f9f6f2", \
                    512:"#f9f6f2", 1024:"#f9f6f2", 2048:"#f9f6f2" }
FONT = ("Verdana", 40, "bold")
STATUS_FONT = ("Verdana", 20, "bold")


class Board(Frame):
    def __init__(self, type):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048 bot')
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        self.grid(sticky="news")
        self.grid_cells = []
        self.score = 0
        self.last_sum = 0
        self.hightest_number = 0
        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()

        self.mainloop()
    
    
    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE/GRID_LEN, height=SIZE/GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

        self.score_lb = Label(self, text="Score: "+str(self.score), relief='sunken', anchor=W)
        self.score_lb.grid(row=5, column=0, columnspan=5, sticky='we')

        self.last_sum_lb = Label(self, text="Sum of last tiles: "+str(self.last_sum), relief='sunken', anchor=W)
        self.last_sum_lb.grid(row=6, column=0, columnspan=5, sticky='we')

        self.hightest_number_lb = Label(self, text="Highest number: "+str(self.hightest_number), relief='sunken', anchor=W)
        self.hightest_number_lb.grid(row=7, column=0, columnspan=5, sticky='we')

        for x in range(4):
            self.grid_columnconfigure(x, weight=1)
        for y in range(5):
            self.grid_rowconfigure(y, weight=1)

    def init_matrix(self):
        self.matrix = []

        for _ in range(4):
            self.matrix.append([0] * 4)
        

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number], fg=CELL_COLOR_DICT[new_number])
        self.update_idletasks()

game = Board('MC')