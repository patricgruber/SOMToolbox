import numpy as np
from visualizations.iVisualization import VisualizationInterface
from controls.controllers import MetroMapController
import panel as pn
import holoviews as hv
from holoviews.streams import Pipe
import bokeh.palettes as colors
import os


class MetroMap2(VisualizationInterface):

    def __init__(self, main):
        self._main = main
        self._controls = MetroMapController(self._calculate, self._main._dim, self._main._component_names,
                                            name='Metro Map visualization 2.0')
        self._raw_solutions = []
        self._snapped_lines = []

    def _activate_controllers(self, ):
        reference = pn.pane.Str(
            "<ul><li><b>Robert Neumayer, Rudolf Mayer, Georg Pölzlbauer, and Andreas Rauber</b> "
            "The metro visualisation of component planes for self-organising maps.\""
            " In Proceedings of the International Joint Conference on Neural Networks (IJCNN'07), "
            "Orlando, FL, USA, August 12-17 2007. IEEE Computer Society.</li>"
            "<li><b>Robert Neumayer, Rudolf Mayer, and Andreas Rauber.</b> "
            "\"Component selection for the metro visualisation of the SOM.\" "
            "In Proceedings of the 6th International Workshop on Self-Organizing Maps (WSOM'07), "
            "Bielefeld, Germany, September 3-6 2007.</li></ul>")
        self._main._controls.append(pn.Column(self._controls, reference))
        self._calculate(calculating=True)

    def _deactivate_controllers(self, ):
        self._main._pipe_paths.send([])
        self._main._pipe.send([])
        self._main._pdmap[0] = self._main._Image.apply.opts(cmap=self._main._maincontrol.param.colormap,
                                                            color_levels=None) * self._main._Paths

    def _calculate(self, calculating):
        if calculating:
            raw_solutions = self._get_centers()
            _snapped_lines = []

            for line in raw_solutions:
                _snapped_lines.append(self._find_snapped_line(line))

            self._raw_solutions = self._rotate(np.array(raw_solutions))
            self._snapped_lines = self._rotate(np.array(_snapped_lines))

        lines = None
        if self._controls.snapping:
            lines = self._snapped_lines[self._controls.components_int]
        else:
            lines = self._raw_solutions[self._controls.components_int]

        overlay = []
        data = hv.Image(UMatrix(self._main._m, self._main._n, self._main._weights, self._main._dim)) \
            .apply \
            .opts(
            cmap=self._main._maincontrol.param.colormap,
            xlim=self._main._xlim,
            ylim=self._main._ylim,
            width=self._main._width,
            height=self._main._height
        )

        overlay.append(data)
        for i, pts in enumerate(lines):
            c = colors.Category20[20][self._controls.components_int[i]]
            overlay.append(hv.Path(np.array(pts)).opts(color=c, line_width=4))
            overlay.append(hv.Points(np.array(pts)).opts(fill_color=c, color='black', size=15))

            label = self._main._component_names[self._controls.components_int[i]]

            end, start = pts[0], pts[-1]
            if self._controls.labelling:
                # endpoint labels
                overlay.append(hv.Text(start[0], start[1], "+", 15))
                overlay.append(hv.Text(end[0], end[1], "-", 15))

                # attribute label shadow
                overlay.append(hv.Text(start[0] + 0.028, start[1] - 0.003, label, 12, halign='left').opts(color='black'))
                # attribute label
                overlay.append(hv.Text(start[0] + 0.025, start[1], label, 12, halign='left').opts(color=c))

        self._main._pdmap[0] = hv.Overlay(overlay).collate()

    def _rotate(self, lines):
        lines[:, :, [0, 1]] = lines[:, :, [1, 0]]
        scale = lambda a, b, x, minx, maxx: (b - a) * ((x - minx) / (maxx - minx)) + a
        lines[:, :, 0] = scale(-0.5, 0.5, lines[:, :, 0], -0.5, self._main._n)
        lines[:, :, 1] = scale(-0.5, 0.5, lines[:, :, 1], -0.5, self._main._m)
        lines[:, :, [1]] = -1 * lines[:, :, [1]]
        return lines

    def _digitize(self, component):
        raw = self._main._weights[:, component].reshape(self._main._m, self._main._n)
        ranges = np.linspace(raw.min(), raw.max(), self._controls.stops + 1)
        return np.digitize(raw, ranges[:-1])

    def _get_centers(self, ):
        lines = np.array([])
        for component in range(self._main._dim):
            binned = self._digitize(component)
            centers = []
            for i in range(1, self._controls.stops + 1):
                match = np.argwhere(binned == i)
                centers.append(np.sum(match, axis=0) / match.shape[0])

            if len(lines) == 0:
                lines = [np.array(centers)]
            else:
                lines = np.vstack((lines, [np.array(centers)]))
        return lines

    def _find_snapped_line(self, line):

        lineSegmentForward, lineSegmentBackward = [], []
        length = len(line)
        snapped_lines = np.array([])
        for i, pt in enumerate(line):
            for point in self._get_neighbours(pt.tolist()):
                if i >= 1:
                    if i < length: lineSegmentForward = self._doSnapping(point, i, 1, line)
                    if i > 0:      lineSegmentBackward = self._doSnapping(point, i, -1, line)
                    if i > 0 and i < length:
                        for j in range(length):
                            if (lineSegmentForward[0][j] == -1): lineSegmentForward[0][j] = lineSegmentBackward[0][j]
                            if (lineSegmentBackward[0][j] == -1): lineSegmentBackward[0][j] = lineSegmentForward[0][j]
                    if i == length - 1: lineSegmentForward = lineSegmentBackward

                    lineSegmentForward[0][i] = point

                    if len(snapped_lines) == 0:
                        snapped_lines = [np.array(lineSegmentForward[0])]
                    else:
                        snapped_lines = np.vstack((snapped_lines, [np.array(lineSegmentForward[0])]))

        return self._get_closest_snappedline(snapped_lines, line)

    def _doSnapping(self, point, position, direction, line):
        snapped_line = [-1] * len(line)
        step = position
        while step + direction >= 0 and step + direction < len(line):
            step = step + direction
            point = self._get_closest_points(point, line[step])
            snapped_line[step] = point
        return [snapped_line]

    def _get_closest_snappedline(self, lines, original):
        total_distance = np.inf
        length = len(original)
        dist = lambda x, y: np.linalg.norm(x - y)
        minimal = None
        for line in lines:
            distance1, distance2 = 0, 0
            for i in range(length):
                distance1 += dist(line[i], original[i])
                distance2 += dist(line[i], original[length - i - 1])
            if total_distance > min(distance1, distance2):
                total_distance = min(distance1, distance2)
                minimal = line
        return minimal

    def _get_closest_points(self, pt1, pt2):

        vertical = [[pt1[0], i] for i in range(self._main._m)]
        horizontal = [[i, pt1[1]] for i in range(self._main._n)]

        points = horizontal + vertical
        for i in range(1, max(self._main._m, self._main._n)):
            if pt1[0] + i < self._main._m and pt1[1] + i < self._main._n:
                points.append([pt1[0] + i, pt1[1] + i])
            if pt1[0] - i >= 0 and pt1[1] - i >= 0: points.append([pt1[0] - i, pt1[1] - i])

            if pt1[0] + i < self._main._m and pt1[1] - i >= 0: points.append([pt1[0] + i, pt1[1] - i])
            if pt1[0] - i >= 0 and pt1[1] + i < self._main._n: points.append([pt1[0] - i, pt1[1] + i])

        distances = np.linalg.norm(np.array(points)[:, None, :] - pt2, axis=-1)
        distances = np.column_stack((points, distances))
        distances = distances[distances[:, -1].argsort()]

        return distances[0, 0:2].tolist()

    def _get_neighbours(self, pt):
        pt = np.floor(pt).tolist()
        neighbours = [pt]
        if pt[1] + 1 < self._main._m: neighbours.append([pt[0], pt[1] + 1])
        if pt[0] + 1 < self._main._n: neighbours.append([pt[0] + 1, pt[1]])
        if pt[0] + 1 < self._main._n and pt[1] + 1 < self._main._m:
            neighbours.append([pt[0] + 1, pt[1] + 1])
        return neighbours


# U-Matrix - implementation
def UMatrix(_m, _n, _weights, _dim):
    U = _weights.reshape(_m, _n, _dim)
    U = np.insert(U, np.arange(1, _n), values=0, axis=1)
    U = np.insert(U, np.arange(1, _m), values=0, axis=0)
    # calculate interpolation
    for i in range(U.shape[0]):
        if i % 2 == 0:
            for j in range(1, U.shape[1], 2):
                U[i, j][0] = np.linalg.norm(U[i, j - 1] - U[i, j + 1], axis=-1)
        else:
            for j in range(U.shape[1]):
                if j % 2 == 0:
                    U[i, j][0] = np.linalg.norm(U[i - 1, j] - U[i + 1, j], axis=-1)
                else:
                    U[i, j][0] = (np.linalg.norm(U[i - 1, j - 1] - U[i + 1, j + 1], axis=-1) + np.linalg.norm(
                        U[i + 1, j - 1] - U[i - 1, j + 1], axis=-1)) / (2 * np.sqrt(2))

    U = np.sum(U, axis=2)  # move from Vector to Scalar

    for i in range(0, U.shape[0], 2):  # count new values
        for j in range(0, U.shape[1], 2):
            region = []
            if j > 0: region.append(U[i][j - 1])  # check left border
            if i > 0: region.append(U[i - 1][j])  # check bottom
            if j < U.shape[1] - 1: region.append(U[i][j + 1])  # check right border
            if i < U.shape[0] - 1: region.append(U[i + 1][j])  # check upper border

            U[i, j] = np.median(region)

    return U
