import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class SimpleGraph():
    """ simple network graphing tool """
    def __init__(self):
        self.vertices = []
        self.edges = []
        return

    def add_vertex(self, name, color='r'):
        self.vertices.append((name, color))
        return

    def add_edge(self, vertex_in, vertex_out, weight=1):
        self.edges.append(((vertex_in, vertex_out), weight))
        return

    def draw(self, save_path):
        num_verticies = len(self.vertices)
        fig, ax = plt.subplots(figsize=(num_verticies * 1 + 2, 3))

        vertex_locs = {}
        circle_radius = 0.1

        for i, (v, c) in enumerate(self.vertices):
            loc = np.array((i, 0.5))
            circle = plt.Circle(loc, circle_radius, color=c)
            ax.add_patch(circle)
            ax.text(*loc + np.array((-0.03, 1.2 * circle_radius)), v)
            vertex_locs[v] = loc

        style = "Simple, tail_width=0.5, head_width=4, head_length=4"
        kw = dict(arrowstyle=style, color="k")
        for ((v_in, v_out), w) in self.edges:
            in_loc, out_loc = vertex_locs[v_in], vertex_locs[v_out]
            arrow = patches.FancyArrowPatch(
                            in_loc + np.array((circle_radius, 0)),
                            out_loc - np.array((circle_radius, 0)),
                            connectionstyle=f"arc3,rad={.5 * w}", **kw)
            ax.add_patch(arrow)

        plt.xlim([-0.5, num_verticies - 0.5])
        plt.ylim([-1, 1])
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return


def visualize_graph(computation_graph, save_path='graph.png'):
    """
    Visualizes computation graph and stores it in `save_path.png`
    Args:
        computation_graph: computation graph constructed from `self.graph` in
            OverparamLinear and OverparamConv2d layers
        save_path (str): save path
    """
    vertices, edges = computation_graph

    graph = SimpleGraph()
    graph.verbose = False

    for i, v in enumerate(vertices):
        if v in ['s', 't']:
            graph.add_vertex(v, color='k')
        else:
            graph.add_vertex(v, color='b' if edges[v]['norm'] else 'r')

    for e in edges.values():
        graph.add_edge(e['in'], e['out'], weight=0)

        for r in e['res']:
            graph.add_edge(r, e['out'], weight=1)

    return graph.draw(save_path)
