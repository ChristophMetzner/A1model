from bokeh.themes import built_in_themes

colorListType = 'alternate'  # 'graded'

if colorListType == 'alternate':
    colorList = [[0.42,0.67,0.84], [0.90,0.76,0.00], [0.42,0.83,0.59], [0.90,0.32,0.00],
            [0.34,0.67,0.67], [0.90,0.59,0.00], [0.42,0.82,0.83], [1.00,0.85,0.00],
            [0.33,0.67,0.47], [1.00,0.38,0.60], [0.57,0.67,0.33], [0.5,0.2,0.0],
            [0.71,0.82,0.41], [0.0,0.2,0.5], [0.70,0.32,0.10]]*3
elif colorListType == 'graded' and __gui__:
    import matplotlib
    cmap = matplotlib.cm.get_cmap('jet')
    colorList = [cmap(x) for x in np.linspace(0,1,12)]
