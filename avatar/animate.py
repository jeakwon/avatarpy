import plotly.graph_objects as go

class Animate:
    def __init__(self, parent, visible=['head', 'fbody', 'hbody', 'tail', 'rarm', 'larm', 'rleg', 'lleg']):
        self.__parent = parent
        self.visible = visible
        self._figure = None
    
    def __repr__(self):
        return f'Animate3D instance of {self.__parent}'

    def __call__(self, inidces):
        return self.make(inidces)
        
    def make(self, indices):
        self._figure = self.get_figure(indices)
        return self
    
    def show(self):
        self._figure.show()
        return self
    
    def save(self, html):
        self._figure.write_html(html)
        return self
        
    def get_frames(self, indices):
        frames = []
        for idx in indices:
            data=[]
            for name in self.__parent.vectors.keys():
                vector_coords = self.__parent.get_vector_coords(name)
                data.append(
                    go.Scatter3d(
                        x = vector_coords['x'].loc[idx],
                        y = vector_coords['y'].loc[idx],
                        z = vector_coords['z'].loc[idx],
                        name=name,
                        mode="lines",
                        visible="legendonly" if name not in self.visible else None,
                    )
                )
            frame = go.Frame(data=data, name=idx)
            frames.append(frame)
        return frames
    
    def get_steps(self, indices):
        steps=[]
        for idx in indices:
            step = dict(
                label=idx,
                method="animate",
                args=[
                    [idx],
                    dict(
                        frame=dict(duration=0, redraw=True),
                        mode="immediate",
                        transition=dict(duration=0)
                    ),
                ],
            )
            steps.append(step)
        return steps
    
    def get_buttons(self):
        play_button = dict(
            label="Play",
            method="animate",
            args=[
                None,
                dict(
                    frame=dict(duration=0, redraw=True),
                    fromcurrent=True,
                    transition=dict(duration=0),
                )
            ],
        )
        pause_button = dict(
            label="Pause",
            method="animate",
            args=[
                [None],
                dict(
                    frame=dict(duration=0, redraw=True),
                    mode="immediate",
                    transition=dict(duration=0),
                )
            ],
        )
        return [play_button, pause_button]
    
    def get_updatemenus(self, indices):
        updatemenus=[dict(
            x=0.1,
            y=0,
            xanchor="right",
            yanchor="top",
            pad=dict(r=10, t=87),
            buttons=self.get_buttons(),
            direction="left",
            showactive=False,
            type="buttons",

        )]
        return updatemenus
    
    def get_sliders(self, indices):
        sliders = [dict(
            x=0.1,
            y=0,
            xanchor="left",
            yanchor="top",
            pad=dict(b=10, t=50),
            len=0.9,
            active=0,
            currentvalue=dict(
                font=dict(size=20),
                prefix="Time:",
                visible=True,
                xanchor="right",
            ),
            transition=dict(duration=0),
            steps=self.get_steps(indices),
        )]
        return sliders
    
    def get_layout(self, indices):
        layout = go.Layout(
            width=600,
            height=600,
            scene=dict(
                    xaxis=dict(range=[self.__parent.x_min, self.__parent.x_max], autorange=False),
                    yaxis=dict(range=[self.__parent.y_min, self.__parent.y_max], autorange=False),
                    zaxis=dict(range=[self.__parent.z_min, self.__parent.z_max], autorange=False),
            ),
            scene_aspectmode='cube',
            updatemenus=self.get_updatemenus(indices),
            sliders=self.get_sliders(indices),
        )
        return layout
    
    def get_figure(self, indices, **kwargs):
        frames = self.get_frames(indices)
        layout = self.get_layout(indices)
        data = frames[0]['data']
        figure = go.Figure(
            data=data,
            layout=layout,
            frames=frames,
        )
        return figure