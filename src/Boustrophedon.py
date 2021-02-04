import matplotlib.pyplot as plt
from sympy import Polygon, Line, Line2D, Point2D, Segment2D, Segment, convex_hull
import shapely.geometry as shp
from seidel.point import Point
import seidel.trapezoidal_map as tz
import numpy as np
import sys
from tsp_solver.greedy import solve_tsp
from functools import cmp_to_key
import pyclipper

first = True


class Poly:
    def __init__(self, points, ecart_fauchee, R, compute=False):
        self.p = Polygon(*points)
        self.d = ecart_fauchee
        self.r = R
        self.angles = self.compute_angles()
        if self.p.area < self.d * 2 * self.r:
            self.too_tiny = True
        else:
            self.too_tiny = False
            if self.p.is_convex() and compute:
                self.a = self.best_orientation()
                self.waypoints = self.compute_lines()

    def __str__(self):
        res = '['
        for p in self.p.vertices:
            res += '(' + str(p.x) + ',' + str(p.y) + ')'

        return res + ']'

    def compute_angles(self):
        angles = []
        n = len(self.p.vertices)
        cw = self.p._isright(self.p.vertices[-1], self.p.vertices[0], self.p.vertices[1])

        if not cw :
            angles = [float(self.p.angles[v]) for v in self.p.vertices]
        else:
            angles = [2 * np.pi - float(self.p.angles[v]) for v in self.p.vertices]

        return angles

    def best_orientation(self):

        # polygon must not have three points aligned !
        dim = len(self.p.vertices)
        p_a = self.p.vertices.index(min(self.p.vertices, key=lambda p: p.y))
        p_b = self.p.vertices.index(max(self.p.vertices, key=lambda p: p.y))
        rotated_angle = 0
        angle = 0
        max_width = np.inf
        caliper_a = np.array([[1., 0.]])
        caliper_b = np.array([[-1., 0.]])

        while rotated_angle < np.pi:
            #plt.clf()
            #self.draw()
            #plt.plot([self.p.vertices[p_a].x, self.p.vertices[(p_a + 1)%dim].x], [self.p.vertices[p_a].y, self.p.vertices[(p_a + 1)%dim].y], '-b')
            #plt.plot([self.p.vertices[p_b].x, self.p.vertices[(p_b + 1) % dim].x], [self.p.vertices[p_b].y, self.p.vertices[(p_b + 1) % dim].y], '-g')

            edge_a = np.array(self.p.vertices[(p_a + 1)%dim] - self.p.vertices[p_a], dtype=np.float64)
            edge_b = np.array(self.p.vertices[(p_b + 1)%dim] - self.p.vertices[p_b], dtype=np.float64)
            L_u, L_v, ps = np.linalg.norm(edge_a), np.linalg.norm(caliper_a), np.vdot(caliper_a, edge_a)
            angle_a = np.arccos(ps/(L_u*L_v))
            L_u, L_v, ps = np.linalg.norm(edge_b), np.linalg.norm(caliper_b), np.vdot(caliper_b, edge_b)
            angle_b = np.arccos(ps / (L_u * L_v))

            #plt.plot([self.p.vertices[p_a].x, self.p.vertices[p_a].x + caliper_a[0, 0]],
            #         [self.p.vertices[p_a].y, self.p.vertices[p_a].y + caliper_a[0, 1]], '-y')
            #plt.plot([self.p.vertices[p_b].x, self.p.vertices[p_b].x + caliper_b[0, 0]],
            #         [self.p.vertices[p_b].y, self.p.vertices[p_b].y + caliper_b[0, 1]], '-y')

            if angle_a < angle_b:

                L = Line2D(Point2D(self.p.vertices[p_a].x, self.p.vertices[p_a].y),
                         Point2D(self.p.vertices[p_a].x + caliper_a[0, 0], self.p.vertices[p_a].y + caliper_a[0, 1]))
                S = L.perpendicular_segment(self.p.vertices[p_b])
                #plt.plot([S.points[0].x, S.points[1].x], [S.points[0].y, S.points[1].y], '-y')
                width = S.length
                p_a = (p_a + 1) % dim

            else:
                L = Line2D(Point2D(self.p.vertices[p_b].x, self.p.vertices[p_b].y),
                         Point2D(self.p.vertices[p_b].x + caliper_b[0, 0], self.p.vertices[p_b].y + caliper_b[0, 1]))
                S = L.perpendicular_segment(self.p.vertices[p_a])
                #plt.plot([S.points[0].x, S.points[1].x], [S.points[0].y, S.points[1].y], '-y')
                width = S.length
                p_b = (p_b + 1) % dim
            #plt.pause(0.0001)

            if width < max_width:
                max_width = width
                angle = rotated_angle

            theta = min(angle_a, angle_b)
            rotated_angle += theta

            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            caliper_a = (R @ caliper_a.T).T
            caliper_b = (R @ caliper_b.T).T

        return float(angle)

    def compute_lines(self):
        points, inter = [], []
        x_min, y_min, x_max, y_max = self.p.bounds
        reached = False
        covered = False
        r = np.sqrt(self.d ** 2 + 3 ** 2)
        if abs(self.a) <= (np.pi/2.) - 0.01:
            a = np.tan(self.a)
            b = y_max
            while not covered:
                p1, p2 = [(x_min, a * x_min + b), (x_max, a * x_max + b)]
                inter = self.p.intersection(Line(p1, p2))
                if inter:
                    reached = True
                    if len(inter) > 1:
                        p1, p2 = inter
                        points.extend([p1, p2])
                    if isinstance(inter[0], Segment2D):
                        p1, p2 = inter[0].points
                        if p1.x > p2.x:
                            p1, p2 = p2, p1
                        points.extend([p1, p2])
                elif reached:
                    covered = True
                b -= self.d / np.sin(0.5 * np.pi - self.a)
        else:
            b = 0
            while not covered:
                p1, p2 = [(x_min + b, y_min), (x_min + b, y_max)]
                inter = self.p.intersection(Line(p1, p2))
                if inter:
                    reached = True
                    if len(inter) > 1:
                        p1, p2 = inter
                        points.extend([p1, p2])
                    if isinstance(inter[0], Segment2D):
                        p1, p2 = inter[0].points
                        if p1.y > p2.y:
                            p1, p2 = p2, p1
                        points.extend([p1, p2])
                elif reached:
                    covered = True
                b += self.d

        # Si il y a des points en première et dernière position, ils ne feront pas partie d'une ligne, on les elimine
        dim = len(points)

        # on range les points dans l'ordre du boustrophedon
        if (self.a != np.pi / 2 and points[0].x > points[1].x) or (self.a == np.pi / 2 and points[0].y > points[1].y):
            points[0], points[1] = points[1], points[0]
        n_k = dim // 4
        for i in range(n_k):
            points[2 + 4 * i], points[3 + 4 * i] = points[3 + 4 * i], points[2 + 4 * i]
        return points

    def classify(self, obstacle=False):
        classes = []
        n = len(self.p.vertices)
        #print(self.p.vertices)
        #print([self.angles[i] * 180 / np.pi for i in range(len(self.angles))])

        for i in range(n):
            v = self.p.vertices[i]
            v_prev = self.p.vertices[i-1]
            v_next = self.p.vertices[(i+1)%n]
            angle = self.angles[i]

            if (v.x <= v_prev.x and v.x <= v_next.x):
                if angle < np.pi:
                    if not obstacle:
                        classes.append('OPEN')
                    else:
                        classes.append('SPLIT')
                else:
                    if not obstacle:
                        classes.append('SPLIT')
                    else:
                        classes.append('OPEN')
            elif (v.x >= v_prev.x and v.x >= v_next.x):
                if angle < np.pi:
                    if not obstacle:
                        classes.append('CLOSE')
                    else:
                        classes.append('MERGE')
                else:
                    if not obstacle:
                        classes.append('MERGE')
                    else:
                        classes.append('CLOSE')
            elif (v.x <= v_prev.x and v.x >= v_next.x):
                if angle < np.pi:
                    if not obstacle:
                        classes.append('CEIL_CONVEX')
                    else:
                        classes.append('FLOOR_CONCAVE')
                else:
                    if not obstacle:
                        classes.append('CEIL_CONCAVE')
                    else:
                        classes.append('FLOOR_CONVEXE')
            elif (v.x >= v_prev.x and v.x <= v_next.x):
                if angle < np.pi:
                    if not obstacle:
                        classes.append('FLOOR_CONVEX')
                    else:
                        classes.append('CEIL_CONCAVE')
                else:
                    if not obstacle:
                        classes.append('FLOOR_CONCAVE')
                    else:
                        classes.append('CEIL_CONVEX')
            else:
                raise Exception("Unusual issue occured whille classing vertices")

        return classes

    def draw(self, color='k'):
        for i in range(len(self.p.vertices) - 1):
            plt.plot([self.p.vertices[i].x, self.p.vertices[i+1].x], [self.p.vertices[i].y, self.p.vertices[i+1].y], color=color)
        plt.plot([self.p.vertices[len(self.p.vertices) - 1].x, self.p.vertices[0].x], [self.p.vertices[len(self.p.vertices) - 1].y, self.p.vertices[0].y], color=color)


class Cell:
    def __init__(self):
        self.Lc = []
        self.Lf = []
        self.closed = False

    def __eq__(self, other):
        return self.Lc == other.Lc and self.Lf == other.Lf


class Boustrophedon:
    def __init__(self, boundary_points, ecart_fauchee, l_rope=8, secu=1, obstacles=[]):
        #print("outer: ", boundary_points)
        #print("obs: ", obstacles)
        self.d = ecart_fauchee
        self.r = l_rope
        self.secu = secu
        self.outer_raw = Poly(boundary_points, ecart_fauchee, self.r+self.secu)
        self.obstacles_raw = [Poly(obstacle, ecart_fauchee, self.r+self.secu) for obstacle in obstacles]
        self.corrected_zone()
        self.Lcll = []
        self.Lp = []

    def corrected_zone(self):
        def contract_expend(area, contract):
            if contract:
                alpha = -1
            else:
                alpha = 1
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(area.p.vertices, pyclipper.JT_SQUARE, pyclipper.ET_CLOSEDPOLYGON)
            clip_poly = pco.Execute(alpha * (area.r))

            if len(clip_poly) > 1:
                raise Exception("La réduction d'espace liée à la corde / sécurité est trop grande !")

            return tuple(clip_poly[0])

        self.outer = Poly(contract_expend(self.outer_raw, True), self.d, self.r+self.secu)
        self.obstacles = [Poly(contract_expend(obstacle, False), self.d, self.r+self.secu) for obstacle in self.obstacles_raw]

        for o in range(len(self.obstacles)-1, -1, -1):
            obstacle = self.obstacles[o]
            if self.outer.p.intersect(obstacle.p):
                outer_points = list(map(lambda p: (float(p.x), float(p.y)), self.outer.p.vertices))
                obstacle_points = list(map(lambda p: (float(p.x), float(p.y)), obstacle.p.vertices))
                p_outer = shp.polygon.orient(shp.Polygon(outer_points))
                p_obs = shp.polygon.orient(shp.Polygon(obstacle_points))
                new_outer = shp.polygon.orient(p_outer.difference(p_outer.intersection(p_obs)))
                new_outer_points = tuple(new_outer.exterior.coords)
                self.outer = Poly(new_outer_points, self.d, self.r+self.secu)
                self.obstacles.remove(obstacle)

    def decomposition(self):
        def list_comp(a, b):
            if a[0].x != b[0].x:
                if a[0].x > b[0].x:
                    return 1
                elif a[0].x < b[0].x:
                    return -1
            else:
                if a[0].y != b[0].y:
                    if a[0].y > b[0].y:
                        return 1
                    elif a[0].y < b[0].y:
                        return -1
                else:
                    return 0
        def find_cell_close(v, Lcll, Lv):
            candidates = []

            for c_ind in range(len(Lcll)):
                c = Lcll[c_ind]
                if not c.closed:
                    candidates.append((c, c_ind))
            if len(candidates) == 1:
                return candidates[0][1]
            else:
                v_candidates = []
                for elmt in Lv:
                    if (v.x <= elmt[0].x) and (elmt[1] == "CLOSE" or elmt[1] == "CLOSE_MOD"):
                        v_candidates.append(elmt[0])
                v_candidates.sort(key=lambda p: p.y)
                candidates.sort(key=lambda c: c[0].Lc[-1].y)
                index = v_candidates.index(v)
                return candidates[index][1]

        def find_cell_merge(v, Lcll):
            candidates = []
            for c_ind in range(len(Lcll)):
                c = Lcll[c_ind]
                if not c.closed:
                    candidates.append((c, c_ind))

            candidates.sort(key=lambda c: c[0].Lf[-1].y)

            if len(candidates) < 2:
                raise Exception("Error: pas assez de candidats")
            elif len(candidates) == 2:
                return candidates[0][1], candidates[1][1]
            else:
                for i in range(len(candidates)-1):
                    c_down, c_up = candidates[i], candidates[i+1]
                    if c_down[0].Lf:
                        if c_up[0].Lc:
                            if v.y >= c_down[0].Lf[-1].y and v.y <= c_up[0].Lc[-1].y:
                                return c_down[1], c_up[1]
                        else:
                            if v.y >= c_down[0].Lf[-1].y:
                                return c_down[1], c_up[1]

            raise Exception("Error: cells non trouvées")

        def find_cell(vf, vc, Lcll):
            candidates = []

            for c_ind in range(len(Lcll)):
                c = Lcll[c_ind]
                if not c.closed:
                    candidates.append((c, c_ind))
            if len(candidates) == 1:
                return candidates[0][1]
            else:
                for c in candidates:
                    if c[0].Lf and c[0].Lc:
                        if vf.y <= c[0].Lc[-1].y and vc.y >= c[0].Lf[-1].y:
                            return c[1]
                    elif c[0].Lf:
                        if vc.y >= c[0].Lf[-1].y:
                            return c[1]
                    else:
                        if vf.y <= c[0].Lc[-1].y:
                            return c[1]

            raise Exception("Error: cell non trouvée !")

        ymin, ymax = self.outer.p.bounds[1], self.outer.p.bounds[3]
        Le = []
        Lcll = []

        classes = self.outer.classify()
        Lv = [[self.outer.p.vertices[i], classes[i], 0, i] for i in range(len(self.outer.p.vertices))]

        for i in range(len(self.obstacles)):
            obstacle = self.obstacles[i]
            classes = obstacle.classify(True)
            Lv.extend([[obstacle.p.vertices[j], classes[j], i+1, j] for j in range(len(obstacle.p.vertices))])

        fct_sort = cmp_to_key(list_comp)
        Lv.sort(key=fct_sort)

        for i in range(len(Lv)-1, 0, -1):
            if Lv[i-1][2] == 0:
                P1, P2 = self.outer.p.vertices[Lv[i-1][3]-1], self.outer.p.vertices[(Lv[i-1][3]+1)%len(self.outer.p.vertices)]
            else:
                P1, P2 = self.obstacles[Lv[i-1][2]-1].p.vertices[Lv[i - 1][3] - 1], self.obstacles[Lv[i-1][2]-1].p.vertices[(Lv[i - 1][3] + 1) % len(self.obstacles[Lv[i-1][2]-1].p.vertices)]

            if Lv[i][0] == P1 or Lv[i][0] == P2:
                if (Lv[i][1] == 'OPEN' and Lv[i-1][1] == 'OPEN') and (Lv[i][0].x == Lv[i-1][0].x):
                    Lv[i - 1][1] = 'OPEN_MOD'
                    if Lv[i][0].y < Lv[i-1][0].y:
                        Lv[i][1] = 'FLOOR_CONVEX'
                    else:
                        Lv[i][1] = 'CEIL_CONVEX'
                elif (Lv[i][1] == 'CLOSE' and Lv[i-1][1] == 'CLOSE') and (Lv[i][0].x == Lv[i-1][0].x):
                    Lv[i][1] = 'CLOSE_MOD'
                    if Lv[i][0].y > Lv[i - 1][0].y:
                        Lv[i-1][1] = 'FLOOR_CONVEX'
                    else:
                        Lv[i-1][1] = 'CEIL_CONVEX'
                elif (Lv[i][1] == 'SPLIT' and Lv[i-1][1] == 'SPLIT') and (Lv[i][0].x == Lv[i-1][0].x):
                    Lv[i - 1][1] = 'SPLIT_MOD'
                    Lv[i - 1].append(Lv[i][0])
                    if Lv[i][0].y > Lv[i - 1][0].y:
                        Lv[i][1] = 'FLOOR_CONVEX'
                    else:
                        Lv[i][1] = 'CEIL_CONVEX'
                elif (Lv[i][1] == 'MERGE' and Lv[i-1][1] == 'MERGE') and (Lv[i][0].x == Lv[i-1][0].x):
                    Lv[i][1] = 'MERGE_MOD'
                    Lv[i].append(Lv[i-1][0])
                    if Lv[i][0].y < Lv[i - 1][0].y:
                        Lv[i - 1][1] = 'FLOOR_CONVEX'
                    else:
                        Lv[i - 1][1] = 'CEIL_CONVEX'
        print("--- AREA ---")
        print(self.outer)
        print("--- OBSTACLES ---")
        for obs in self.obstacles:
            print(obs)
        print("------")
        while Lv:
            v = Lv[0]
            sweep_line = Segment2D((v[0].x, ymin), (v[0].x, ymax))

            if v[2] == 0:
                index = self.outer.p.vertices.index(v[0])
                v_prev = self.outer.p.vertices[index - 1]
                v_next = self.outer.p.vertices[(index + 1)%len(self.outer.p.vertices)]
                edge_left = Segment2D(v_prev, v[0])
                edge_right = Segment2D(v[0], v_next)
            else:
                index = self.obstacles[v[2]-1].p.vertices.index(v[0])
                v_prev = self.obstacles[v[2]-1].p.vertices[index - 1]
                v_next = self.obstacles[v[2]-1].p.vertices[(index + 1) % len(self.obstacles[v[2]-1].p.vertices)]
                edge_left = Segment2D(v[0], v_next)
                edge_right = Segment2D(v_prev, v[0])

            if v[1] == 'OPEN' or v[1] == 'OPEN_MOD':
                Le.extend([edge_left, edge_right])
                c = Cell()
                c.Lf.append(v[0])
                Lcll.append(c)

            elif v[1] == 'SPLIT' or v[1] == 'SPLIT_MOD':
                Lv_inter = list(filter(([]).__ne__, [sweep_line.intersection(edge) for edge in Le]))
                for k in range(len(Lv_inter) - 1, -1, -1):
                    if isinstance(Lv_inter[k][0], Segment2D):
                        tmp = Lv_inter[k][0]
                        Lv_inter.remove(Lv_inter[k])
                        Lv_inter.insert(k, [tmp.points[0]])
                        Lv_inter.insert(k+1, [tmp.points[1]])
                Lv_inter.sort(key=lambda x:x[0].y)
                vf, vc = Lv_inter[0][0], Lv_inter[-1][0]
                for v_inter in Lv_inter:
                    v_inter = v_inter[0]
                    if vf.y < v_inter.y < v[0].y:
                        vf = v_inter
                    if v[0].y < v_inter.y < vc.y:
                        vc = v_inter
                c = Lcll[find_cell(vf, vc, Lcll)]
                c.Lf.append(vf)
                c.Lc.append(vc)
                c.closed = True

                cf, cc = Cell(), Cell()
                if len(v)==5:
                    if v[0].y < v[4].y:
                        cf.Lc.append(v[0])
                    else:
                        cc.Lf.append(v[0])
                else:
                    cf.Lc.append(v[0])
                    cc.Lf.append(v[0])
                cf.Lf.append(vf)
                cc.Lc.append(vc)
                Lcll.extend([cc, cf])

                Le.extend([edge_left, edge_right])

            elif v[1] == 'CEIL_CONVEX':
                Le.remove(edge_right)
                Le.append(edge_left)

                Lv_inter = list(filter(([]).__ne__, [sweep_line.intersection(edge) for edge in Le]))
                for k in range(len(Lv_inter) - 1, -1, -1):
                    if isinstance(Lv_inter[k][0], Segment2D):
                        tmp = Lv_inter[k][0]
                        Lv_inter.remove(Lv_inter[k])
                        Lv_inter.insert(k, [tmp.points[0]])
                        Lv_inter.insert(k+1, [tmp.points[1]])

                Lv_inter.sort(key=lambda x: x[0].y)
                vf = Lv_inter[0][0]
                for v_inter in Lv_inter:
                    v_inter = v_inter[0]
                    if vf.y < v_inter.y < v[0].y:
                        vf = v_inter
                c = Lcll[find_cell(vf, v[0], Lcll)]
                c.Lc.append(v[0])

            elif v[1] == 'FLOOR_CONVEX':
                Le.remove(edge_left)
                Le.append(edge_right)

                Lv_inter = list(filter(([]).__ne__, [sweep_line.intersection(edge) for edge in Le]))
                for k in range(len(Lv_inter) - 1, -1, -1):
                    if isinstance(Lv_inter[k][0], Segment2D):
                        tmp = Lv_inter[k][0]
                        Lv_inter.remove(Lv_inter[k])
                        Lv_inter.insert(k, [tmp.points[0]])
                        Lv_inter.insert(k+1, [tmp.points[1]])

                Lv_inter.sort(key=lambda x: x[0].y)
                vc = Lv_inter[-1][0]
                for v_inter in Lv_inter:
                    v_inter = v_inter[0]
                    if v[0].y < v_inter.y < vc.y:
                        vc = v_inter
                c = Lcll[find_cell(v[0], vc, Lcll)]
                c.Lf.append(v[0])

            elif v[1] == 'CEIL_CONCAVE':
                Le.remove(edge_right)
                Le.append(edge_left)

                Lv_inter = list(filter(([]).__ne__, [sweep_line.intersection(edge) for edge in Le]))
                for k in range(len(Lv_inter) - 1, -1, -1):
                    if isinstance(Lv_inter[k][0], Segment2D):
                        tmp = Lv_inter[k][0]
                        Lv_inter.remove(Lv_inter[k])
                        Lv_inter.insert(k, [tmp.points[0]])
                        Lv_inter.insert(k+1, [tmp.points[1]])

                Lv_inter.sort(key=lambda x: x[0].y)
                vf= Lv_inter[0][0]
                for v_inter in Lv_inter:
                    v_inter = v_inter[0]
                    if vf.y < v_inter.y < v[0].y:
                        vf = v_inter
                c = Lcll[find_cell(vf, v[0], Lcll)]
                c.Lc.append(v[0])
                c.Lf.append(vf)
                c.closed = True
                c_new = Cell()
                c_new.Lc.append(v[0])
                c_new.Lf.append(vf)
                Lcll.append(c_new)

            elif v[1] == 'FLOOR_CONCAVE':
                Le.remove(edge_left)
                Le.append(edge_right)

                Lv_inter = list(filter(([]).__ne__, [sweep_line.intersection(edge) for edge in Le]))
                for k in range(len(Lv_inter) - 1, -1, -1):
                    if isinstance(Lv_inter[k][0], Segment2D):
                        tmp = Lv_inter[k][0]
                        Lv_inter.remove(Lv_inter[k])
                        Lv_inter.insert(k, [tmp.points[0]])
                        Lv_inter.insert(k+1, [tmp.points[1]])

                Lv_inter.sort(key=lambda x: x[0].y)
                vc = Lv_inter[-1][0]
                for v_inter in Lv_inter:
                    v_inter = v_inter[0]
                    if v[0].y < v_inter.y < vc.y:
                        vc = v_inter
                c = Lcll[find_cell(v[0], vc, Lcll)]
                c.Lf.append(v[0])
                c.Lc.append(vc)
                c.closed = True
                c_new = Cell()
                c_new.Lf.append(v[0])
                c_new.Lc.append(vc)
                Lcll.append(c_new)

            elif v[1] == 'MERGE' or v[1] == 'MERGE_MOD':
                Le.remove(edge_left)
                Le.remove(edge_right)

                Lv_inter = list(filter(([]).__ne__, [sweep_line.intersection(edge) for edge in Le]))
                for k in range(len(Lv_inter) - 1, -1, -1):
                    if isinstance(Lv_inter[k][0], Segment2D):
                        tmp = Lv_inter[k][0]
                        Lv_inter.remove(Lv_inter[k])
                        Lv_inter.insert(k, [tmp.points[0]])
                        Lv_inter.insert(k+1, [tmp.points[1]])

                Lv_inter.sort(key=lambda x: x[0].y)
                vf, vc = Lv_inter[0][0], Lv_inter[-1][0]
                for v_inter in Lv_inter:
                    v_inter = v_inter[0]
                    if vf.y < v_inter.y < v[0].y:
                        vf = v_inter
                    if v[0].y < v_inter.y < vc.y:
                        vc = v_inter

                if len(v)==5:
                    i_f, i_c = find_cell_merge(v[0], Lcll)
                    cf, cc = Lcll[i_f], Lcll[i_c]
                    if v[0].y < v[4].y:
                        cf.Lc.append(v[0])
                    else:
                        cc.Lf.append(v[0])
                    cf.Lf.append(vf)
                    cc.Lc.append(vc)
                else:
                    i_f, i_c = find_cell_merge(v[0], Lcll)
                    cf, cc = Lcll[i_f], Lcll[i_c]
                    cf.Lc.append(v[0])
                    cc.Lf.append(v[0])
                    cf.Lf.append(vf)
                    cc.Lc.append(vc)

                cc.closed = True
                cf.closed = True
                c_new = Cell()
                c_new.Lc.append(vc)
                c_new.Lf.append(vf)
                Lcll.append(c_new)

            elif v[1] == 'CLOSE' or v[1] == 'CLOSE_MOD':
                Le.remove(edge_left)
                Le.remove(edge_right)
                c = Lcll[find_cell_close(v[0], Lcll, Lv)]
                c.Lf.append(v[0])
                c.closed = True

            else:
                raise Exception("Incorrect class encountered: " + v[1])

            Lv.remove(v)
            # debug
            # print(v[1])
            # for c in Lcll:
            #     lcc = []
            #     for cc in c.Lc:
            #         lcc.append((float(cc.x), float(cc.y)))
            #     lcf = []
            #     for cf in c.Lf:
            #         lcf.append((float(cf.x), float(cf.y)))
            #     print(lcf, lcc)
            # print("---")

        self.Lcll = Lcll
        self.Lp = [Poly(cell.Lf + cell.Lc[::-1], self.d, self.r+self.secu, True) for cell in Lcll]
        for i in range(len(self.Lp)-1, -1, -1):
            if self.Lp[i].too_tiny:
                self.Lp.remove(self.Lp[i])
                self.Lcll.remove(self.Lcll[i])

    def fuse_cells(self, epsilon=1):
        fused_cells = []

        for i in range(len(self.Lp)):
            for j in range(i+1, len(self.Lp)):
                if i!=j:
                    inter = list(set(self.Lp[i].p.vertices) & set(self.Lp[j].p.vertices))
                    for k in range(len(inter)-1, -1, -1):
                        if isinstance(inter[k], Segment2D):
                            tmp = inter[k]
                            inter.remove(inter[k])
                            inter.extend(list(tmp.points))
                    if len(inter) > 1:
                        if set(inter).issubset(set(self.Lp[i].p.vertices)) and set(inter).issubset(set(self.Lp[j].p.vertices)):
                            if self.Lp[j].a - epsilon * np.pi / 180. < self.Lp[i].a < self.Lp[j].a + epsilon * np.pi / 180.:

                                i_belongs = False
                                for test_i in fused_cells:
                                    if i in test_i and j not in test_i:
                                        i_belongs = True
                                        j_belongs = False
                                        for test_j in fused_cells:
                                            if j in test_j:
                                                j_belongs = True
                                                elmt = test_i + test_j
                                                fused_cells.remove(test_i)
                                                fused_cells.remove(test_j)
                                                fused_cells.append(elmt)
                                                break
                                        if not j_belongs:
                                            test_i.append(j)
                                if not i_belongs:
                                    j_belongs = False
                                    for test_j in fused_cells:
                                        if j in test_j:
                                            j_belongs = True
                                            elmt = [i] + test_j
                                            fused_cells.remove(test_j)
                                            fused_cells.append(elmt)
                                            break
                                    if not j_belongs:
                                        fused_cells.append([i, j])

        new_cells = []
        old_cells = []
        info = []
        for i in range(len(fused_cells)):
            c = Cell()
            a = 0
            for index in fused_cells[i]:
                a += self.Lp[index].a
                c.Lf.extend(self.Lcll[index].Lf)
                c.Lc.extend(self.Lcll[index].Lc)
                old_cells.append(self.Lcll[index])

            new_cells.append(c)
            info.append(a/len(fused_cells[i]))

        self.Lcll = [elmt for elmt in self.Lcll if elmt not in old_cells]
        self.Lp = [Poly(cell.Lf + cell.Lc[::-1], self.d, self.r+self.secu, True) for cell in self.Lcll]

        for i in range(len(new_cells)):
            self.Lcll.append(new_cells[i])
            self.Lp.append(Poly(new_cells[i].Lf + new_cells[i].Lc[::-1], self.d, self.r+self.secu, True))
            self.Lp[-1].a = info[i]
            self.Lp[-1].waypoints = self.Lp[-1].compute_lines()

    def tsp_planning(self):
        nodes = []
        dist_matrix = []

        for p in self.Lp:
            nodes.append(p.waypoints[0])
            nodes.append(p.waypoints[-1])

        dim = len(nodes)

        for i in range(dim):
            col = []
            for j in range(dim):
                if j == i:
                    col.append(0.)
                elif i%2 == 0 and j == i+1:
                    col.append(-1000.)
                elif i%2 == 1 and j == i-1:
                    col.append(-1000.)
                else:
                    d = 1000 * np.sqrt(float(nodes[j].x - nodes[i].x)**2 + float(nodes[j].y - nodes[i].y)**2)
                    col.append(d)
            dist_matrix.append(col)

        path = solve_tsp(dist_matrix)

        dim = len(self.Lcll)
        self.Lcll = [self.Lcll[i] for i in path if i in range(dim)]
        self.Lp = [self.Lp[i] for i in path if i in range(dim)]

        return [nodes[i] for i in path]

    def obstacle_avoidance(self, w1, w2):

        L = Line2D(w1, w2)
        waypoints = [w1]
        for o in range(len(self.obstacles)):
            obs = convex_hull(self.obstacles[o].p.vertices, polygon=True)
            inter = obs.intersection(L)
            if inter:
                n = len(obs.vertices)
                indexes = []
                for index in range(n):
                    if L.intersection(Line(obs.vertices[index], obs.vertices[(index+1)%n])):
                        indexes.append(index)
                if len(indexes) > 1:
                    if Segment2D(L.points[0], obs.vertices[indexes[0]]).length < Segment2D(L.points[0], obs.vertices[indexes[1]]).length:
                        i1, i2 = indexes
                    else:
                        i2, i1 = indexes

                points = list(map(lambda p: (float(p.x), float(p.y)), obs.vertices))
                if isinstance(inter, Segment2D):
                    pi1, pi2 = inter.points
                else:
                    pi1, pi2 = inter

                p1 = Polygon([pi2] + [points[i] for i in range(i2 + 1, len(points))] + [points[i] for i in range(i1 + 1)] + [pi1])
                p2 = Polygon([pi1] + [points[i] for i in range(i1 + 1, i2 + 1)] + [pi2])

                if p1.perimeter < p2.perimeter:
                    waypoints.extend(p1.vertices[::-1])
                else:
                    waypoints.extend(p2.vertices)
                L = Line2D(waypoints[-1], w2)

        n = len(self.outer.p.vertices)
        indexes = []
        for index in range(n):
            L = Line2D(waypoints[-1], w2)
            if L.intersection(Line(self.outer.p.vertices[index], self.outer.p.vertices[(index + 1) % n])):
                indexes.append(index)
        if len(indexes) > 1:
            for i in range(len(indexes) // 2):
                if Segment2D(L.points[0], self.outer.p.vertices[indexes[i+0]]).length < Segment2D(L.points[0], self.outer.p.vertices[indexes[i+1]]).length:
                    i1, i2 = indexes
                else:
                    i2, i1 = indexes

        if waypoints[0] == waypoints[1]:
            waypoints.remove(waypoints[0])
        if waypoints[-1] != w2:
            waypoints.append(w2)

        return waypoints