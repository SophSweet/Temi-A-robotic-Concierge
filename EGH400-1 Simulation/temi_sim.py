#!/usr/bin/env python3
import pygame, sys, random
import tkinter as tk
from tkinter import messagebox
import textwrap
from collections import deque

# ─── CONFIG ───────────────────────────────────────────
GRID_W, GRID_H = 20, 12      # cells
CELL    = 40                 # pixels per cell
FPS     = 3                  # slow down animation (frames per second)
ENTRY       = (0, 0)         # Person entry
HOME        = (0, 0)         # Temi home
EXIT_POS    = (GRID_W-1, GRID_H-1)
TARGET_DIST = 1.0            # metres
CELL_METERS = 0.5
FOLLOW_TH   = TARGET_DIST / CELL_METERS  # in cells

ARTWORKS = {
    "The Mona Lisa":   {"pos": (16,  3),
                    "info": "The Mona Lisa is a half-length portrait by Leonardo da Vinci, painted c.1503–1506. It’s famous for her enigmatic smile."},
    "The Scream":  {"pos": (18, 10),
                    "info": "The Scream (1893) by Edvard Munch depicts a figure with an agonized expression against a blood-red sky."},
    "Starry Night": {"pos": ( 4,  9),
                    "info": "The Starry Night (1889) by Vincent van Gogh shows a swirling night sky over a small town, evoking emotion through color."},
}

# ─── UTILS ─────────────────────────────────────────────
def grid_to_px(cell):
    x, y = cell
    return x * CELL + CELL//2, y * CELL + CELL//2

def cell_dist(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return (dx*dx + dy*dy)**0.5

# BFS grid pathfinding
def bfs_path(start, goal):
    q = deque([start])
    came = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (cur[0] + dx, cur[1] + dy)
            if 0 <= nb[0] < GRID_W and 0 <= nb[1] < GRID_H and nb not in came:
                came[nb] = cur
                q.append(nb)
    # reconstruct
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = came.get(node)
    return list(reversed(path))

# ─── DIALOG WINDOWS ────────────────────────────────────
def prompt_guidance():
    dlg = tk.Tk(); dlg.withdraw()
    ans = messagebox.askyesno("Guidance", "Would you like me to guide you to an artwork?", parent=dlg)
    dlg.destroy()
    return ans

def prompt_artwork():
    choice = None
    dlg = tk.Tk(); dlg.title("Choose Artwork"); dlg.geometry("300x200")
    def on_select(n):
        nonlocal choice
        choice = n
        dlg.destroy()
    for name in ARTWORKS:
        btn = tk.Button(dlg, text=name, width=20,
                        command=lambda n=name: on_select(n))
        btn.pack(pady=5)
    dlg.mainloop()
    return choice

def show_description(name):
    dlg = tk.Tk(); dlg.withdraw()
    messagebox.showinfo(name, ARTWORKS[name]["info"], parent=dlg)
    dlg.destroy()

# ─── DRAW FUNCTION ────────────────────────────────────
def draw(screen, font, temi, person):
    screen.fill((240,240,240))
    # grid
    for x in range(GRID_W):
        for y in range(GRID_H):
            pygame.draw.rect(screen, (200,200,200), (x*CELL, y*CELL, CELL, CELL), 1)
    # labels
    entry_px = grid_to_px(ENTRY)
    exit_px  = grid_to_px(EXIT_POS)
    screen.blit(font.render("Entry", True, (0,0,0)), (entry_px[0]+10, entry_px[1]-20))
    screen.blit(font.render("Exit", True, (0,0,0)),  (exit_px[0]+10,   exit_px[1]-20))
    # artworks
    for name,info in ARTWORKS.items():
        pos_px = grid_to_px(info["pos"])
        pygame.draw.circle(screen, (180,100,50), pos_px, 12)
        lbl = font.render(name, True, (0,0,0))
        screen.blit(lbl, (pos_px[0]-lbl.get_width()//2, pos_px[1]-20))
    # person & Temi
    person_px = grid_to_px(person); temi_px = grid_to_px(temi)
    pygame.draw.circle(screen,(255,50,50), person_px, 12)
    screen.blit(font.render("Person",True,(0,0,0)), (person_px[0]+10, person_px[1]-20))
    pygame.draw.circle(screen,(0,120,255), temi_px, 14)
    screen.blit(font.render("Temi",True,(0,0,0)),   (temi_px[0]+10,   temi_px[1]-20))
    pygame.display.flip()

# ─── MAIN ─────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_W*CELL, GRID_H*CELL))
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont(None, 20)

    person = list(ENTRY)
    temi   = list(random.choice([(x,y) for x in range(GRID_W) for y in range(GRID_H) if (x,y)!=ENTRY]))

    full_path = bfs_path(tuple(temi), ENTRY)
    approach = full_path[:-1]
    # animate approach
    for step in approach:
        temi[:] = step; draw(screen, font, temi, person); clock.tick(FPS)

    # guidance dialog
    if prompt_guidance():
        while True:
            art = prompt_artwork()
            if not art: break
            art_path = bfs_path(tuple(temi), ARTWORKS[art]["pos"])
            for step in art_path[1:]:
                temi[:] = step
                # person follows
                if cell_dist(temi, person)>FOLLOW_TH:
                    dx,dy=temi[0]-person[0], temi[1]-person[1]
                    person[0]+=dx//abs(dx) if dx else 0
                    person[1]+=dy//abs(dy) if dy else 0
                draw(screen, font, temi, person); clock.tick(FPS)
            show_description(art)
            if not prompt_guidance(): break

    retreat_t = bfs_path(tuple(temi), HOME)
    retreat_p = bfs_path(tuple(person), EXIT_POS)
    length = max(len(retreat_t), len(retreat_p))
    for i in range(1, length):
        if i < len(retreat_t): temi[:] = retreat_t[i]
        if i < len(retreat_p): person[:] = retreat_p[i]
        draw(screen, font, temi, person); clock.tick(FPS)

    # final done popup
    dlg = tk.Tk(); dlg.withdraw()
    messagebox.showinfo("Done", "Session complete.", parent=dlg)
    dlg.destroy()
    pygame.quit(); sys.exit()

if __name__ == "__main__": main()
