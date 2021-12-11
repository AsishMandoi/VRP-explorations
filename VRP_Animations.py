from manim import *
import numpy as np


def generate_vrp_instance(n, frame_width, frame_height, seed=None):
    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Generate VRP instance
    xc = (np.random.rand(n + 1) - 0.5) * frame_width
    yc = (np.random.rand(n + 1) - 0.5) * frame_height
    instance = np.zeros((n + 1, n + 1))
    for ii in range(n + 1):
        for jj in range(ii + 1, n + 1):
            instance[ii, jj] = np.sqrt((xc[ii] - xc[jj]) ** 2 + (yc[ii] - yc[jj]) ** 2)
            instance[jj, ii] = instance[ii, jj]

    # Return output
    return instance, xc, yc


class FQS(Scene):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = 5
        self.k = 3
        self.XC = [-2.3, -4, 1, 4.8, 1]
        self.YC = [-0.4, 0.8, -1.5, 1.5, 2.3]

    def construct(self):
        #instance, xc, yc = generate_vrp_instance(self.n, self.camera.frame_width-1, self.camera.frame_height-1)

        dots = [Dot([x, y, 0]) for x, y in zip(self.XC, self.YC)]
        squares = [Square(0.3, color=BLUE).move_to(dots[0].get_center()) for i in range(self.k)]
        lines = [Line(dots[i].get_center(), dots[i+1].get_center(), stroke_color=GREEN) for i in range(4)]

        for dot in dots:
            self.add(dot)
        self.wait()

        self.play(
            Write(squares[0])
        )

        for i in range(3):
            self.play(squares[0].animate.move_to(dots[i]))
            self.cur_time_step(dots, 0, i, i)

        self.show_distance(dots[2], dots[3])
        self.wait()


    def cur_time_step(self, nodes, vehicle_num, cur_node_num, time_step):
        label_strings = []
        labels = []
        sigma = "\\sum_{j=1}^{N}{x_{" + str(vehicle_num) + ",j," + str(time_step) + "}="
        summation_str = sigma

        for j in range(self.n):
            # Check whether x_ijk = 0 or 1
            if cur_node_num == j:
                activated = 1
                label_color = BLUE
            else:
                activated = 0
                label_color = GREEN

            # Add label 
            label_str = "x_{"+str(vehicle_num)+","+str(j)+","+str(time_step)+"}"
            label_strings.append(label_str)

            summation_str += label_str
            if j != self.n-1:
                summation_str += "+"

            label = MathTex(label_str, "=" + str(activated), substrings_to_isolate=[label_str])
            label.scale(0.75)
            label.next_to(nodes[j], direction = DOWN)
            label.set_color(label_color)
            labels.append(label)

        summation_tex = MathTex(summation_str, substrings_to_isolate = label_strings)
        summation_tex.to_edge(DOWN)

        # PLAY COMMANDS

        self.play(
            Write(labels[cur_node_num][0])
        )
        self.play(
            Write(labels[cur_node_num][1])
        )
        self.wait()

        self.play(
            AnimationGroup(
                *[Write(labels[i][0]) for i in range(self.n) if i != cur_node_num],
                lag_ratio = 0.25
            )
        )
        self.play(
            *[Write(labels[i][1]) for i in range(self.n) if i != cur_node_num]
        )
        self.wait(2)

        self.play(
            AnimationGroup(
                *[TransformMatchingTex(labels[i], summation_tex) for i in range(self.n)],
                lag_ratio = 0.25
            )
        )
        self.wait()

        
        self.play(
            FadeOut(summation_tex)
        )

    
    def show_distance(self, initial_node, final_node):
        brace = BraceBetweenPoints(initial_node.get_center(), final_node.get_center())
        brace_tex = brace.get_tex("C_{i,j}=" + str(self.get_distance(initial_node, final_node)))

        self.play(Write(brace))
        self.play(Write(brace_tex))

    def get_distance(self, initial_node, final_node):

        x_diff = final_node.get_center()[0] - initial_node.get_center()[1]
        y_diff = final_node.get_center()[1] - initial_node.get_center()[1]

        return np.round(np.sqrt(x_diff**2 + y_diff**2), decimals=3)


class SPS(Scene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = 5
        self.k = 3
        self.XC = [-2.3, -4, 1, 4.8, 1]
        self.YC = [-0.4, 0.8, -1.5, 1.5, 2.3]
        self.TSP = [0, 2, 4, 3, 1, 0]

    def construct(self):
        #instance, xc, yc = generate_vrp_instance(self.n, self.camera.frame_width-1, self.camera.frame_height-1)

        nodes = [
            LabeledDot(Tex(str(i), color=BLACK)).move_to([self.XC[i], self.YC[i], 0])
            for i in range(len(self.XC))
        ]

        arrows = [
            Arrow(start = nodes[self.TSP[i]].get_center(), end = nodes[self.TSP[i+1]].get_center(), color=GOLD)
            for i in range(len(self.TSP)-1) 
        ]

        """
        # Add updaters to arrows
        for i in range(len(arrows)):
            arrows[i].add_updater(
                lambda m: m.put_start_and_end_on(
                    nodes[self.TSP[i]].get_center(), nodes[self.TSP[i+1]].get_center()
                )
            )
        """

        # PLAY COMMANDS
            
        for node in nodes:
            self.add(node)
        self.wait()

        for arrow in arrows:
            self.play(Write(arrow))
        self.wait()

        # Rearrange dots
        client_group = VGroup(*[nodes[self.TSP[i]] for i in range(1, len(self.TSP)-1)])
        self.play(
            nodes[0].animate.to_edge(UP, buff=2).set_x(0),
            client_group.animate.arrange(direction=RIGHT, buff = 2),
            *[FadeOut(arrow) for arrow in arrows]
        )
        self.wait()
            
        arrows = [
            Arrow(start = nodes[self.TSP[i]].get_center(), end = nodes[self.TSP[i+1]].get_center(), color=GOLD)
            for i in range(len(self.TSP)-1) 
        ]

        self.play(*[Write(arrow) for arrow in arrows])
        self.wait()

