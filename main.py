import requests
from bs4 import BeautifulSoup
import urllib.parse
import re
import time
import json
from itertools import chain
import plotly
import plotly.graph_objects as go
from igraph import Graph, EdgeSeq
from flask import Flask, render_template, request, session
import secrets


def get_leaves(tree, node=0):
    children = tree.get(node, [])
    if not children:
        return [node]
    leaves = []
    for child in children:
        leaves.extend(get_leaves(tree, child))
    return leaves

def get_dist_from_root(tree, initial_node):
    if initial_node == 0:
        return 0

    current = [key for key, value in tree.items() if initial_node in value][0]
    count = 1
    while current != 0:
        current = [key for key, value in tree.items() if current in value][0]
        count += 1

    return count

def get_unchecked(tree, ids, n):
    tree_vals = set(chain(*[list(tree.keys())] + list(tree.values())))
    return [i for i in tree_vals if get_dist_from_root(tree, i) < n and ids[i][0] == 'UNCHECKED']

rate_limit_string = "Our systems have detected unusual traffic from your computer network. This page checks to see if it's really you sending the requests, and not a robot."

def get_divs(query):
    response = requests.get(query)
    soup = BeautifulSoup(response.text, 'lxml')

    if rate_limit_string in response.text:
        raise Exception("Rate limit exceeded")

    return soup.find_all('div', {'class' : 'gs_r gs_or gs_scl'})

class Paper:
    def __init__(self, title=None, blurb=None, link=None, other_links=None, authors=None, year=None, citations=None):
        self.title = title
        self.blurb = blurb
        self.link = link
        self.other_links = other_links
        self.authors = authors
        if year is not None:
            self.year = int(year)
        else:
            self.year = None
        self.citations = citations

    def __repr__(self):
        return f'{self.title}, {self.year}: {self.authors}'

    def get_from_div(self, div):
        self.title = div.h3.text

        self.link = div.h3.a['href']

        # get pdf link
        if (other_link := div.find('div', {'class' : 'gs_or_ggsm'})):
            self.other_links = other_link.a['href']
        else:
            self.other_links = 'N/A'
        # what happens when has both pdf and html link

        self.authors = div.find('div', {'class' : 'gs_a'}).text.split('\xa0')[0]

        self.year = int([i for i in re.split('-|,| ', div.find('div', {'class' : 'gs_a'}).text) if i.isdigit()][0])

        self.blurb = div.find('div', {'class' : 'gs_rs'}).text

        # self.citations = [i['href'] for i in div.find('div', {'class' : 'gs_fl'}).find_all('a') if 'cites' in i['href']]
        self.citations = [i['href'] for i in div.findAll('a') if '/scholar?cites' in str(i)][0]
        if self.citations:
            self.citations = 'https://scholar.google.com' + self.citations
        else:
            self.citations = 'N/A'

    def get_from_query(self, query):
        query = f'https://scholar.google.com/scholar?q={urllib.parse.quote(query)}'
        divs = get_divs(query)
        if len(divs) == 0:
            raise Exception(f'{query} not found')
            
        self.get_from_div(divs[0])

def get_citations(paper, n):
    if paper.citations is None or paper.citations == 'N/A':
        raise Exception('Paper must have citations link')
    
    divs = get_divs(paper.citations)[:n]
    papers = [Paper() for _ in divs]
    [i.get_from_div(div) for i, div in zip(papers, divs)]

    return papers

def make_annotations(pos, text, M, scalar=2, font_size=10, font_color='rgb(250,250,250)'):
    L = len(pos)
    if len(text) != L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = []
    for k in range(L):
        annotations.append(
            dict(
                text=text[k], # or replace labels with a different list for the text within the circle
                x=pos[k][0], y=scalar * M - pos[k][1],
                xref='x1', yref='y1',
                font=dict(color=font_color, size=font_size),
                showarrow=False
            )
        )
    return annotations

def create_graph(tree, ids, title):
    G = Graph.Tree(1, 1)

    for key, value in tree.items():
        G.add_vertices(len(value))

        G.add_edges([[key, i] for i in value])

    nr_vertices = G.vcount()
    lay = G.layout('auto')

    position = {k: lay[k] for k in range(nr_vertices)}
    Y = [lay[k][1] for k in range(nr_vertices)]
    M = max(Y)

    es = EdgeSeq(G) # sequence of edges
    E = [e.tuple for e in G.es] # list of edges

    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2*M-position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in E:
        Xe+=[position[edge[0]][0],position[edge[1]][0], None]
        Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

    labels = [i[1].title for i in ids.values()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Xe,
                    y=Ye,
                    mode='lines',
                    line=dict(color='rgb(210,210,210)', width=1),
                    hoverinfo='none'
                    ))
    fig.add_trace(go.Scatter(x=Xn,
                    y=Yn,
                    mode='markers',
                    name='bla',
                    marker=dict(symbol='circle-dot',
                                    size=18,
                                    color='#6175c1',    #'#DB4551', # https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python
                                    line=dict(color='rgb(50,50,50)', width=1)
                                    ),
                    text=labels,
                    hoverinfo='text',
                    opacity=0.8
                    ))
    axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                )
    fig.update_layout(title=title,
                annotations=make_annotations(position, [f'<a href="/paper/{i}">     </a>' for i in ids.keys()], M),
                font_size=12,
                showlegend=False,
                xaxis=axis,
                yaxis=axis,
                margin=dict(l=40, r=40, b=85, t=100),
                hovermode='closest',
                plot_bgcolor='rgb(248,248,248)'
                )
    
    return fig

app = Flask(__name__)
app.secret_key = secrets.token_urlsafe(16)

with open('./templates/paper.html', 'r') as f:
    paper_html = f.read()

def generate_paper_html(title, blurb, link, other_links):
    blurb = f'<p class="paper-blurb">{blurb}</p>\n'

    link = f'<a class="paper-link" href="{link}">{link}</a>\n'

    if other_links == 'N/A':
        other_links = '<b>N/A</b>'
    else:
        other_links = f'<a class="alternate-link" href="{other_links}">{other_links}</a>'

    paper_html_re = {
                    "{{ title }}" : title, 
                    "{{ blurb }}" : blurb,
                    "{{ link }}" : link,
                    "{{ other_links }}" : other_links,
                    }

    replacement = dict((re.escape(k), v) for k, v in paper_html_re.items()) 
    pattern = re.compile('|'.join(replacement.keys()))

    return pattern.sub(lambda match: replacement[re.escape(match.group(0))], paper_html)

def generate_graph_html(pub, depth, nodes):
    query_ids = {0 : ['UNCHECKED', pub]}
    # UNCHECKED
    # CITED
    # NOT_CITED
    query_tree = {0 : []}

    while len((to_check := get_unchecked(query_tree, query_ids, depth))) != 0:
        print(f'Entries to check: {to_check}')
        for current_entry in to_check:
            print(f'Checking: {current_entry}')
            time.sleep(1)
            current_pubs = get_citations(query_ids[current_entry][1], nodes)

            for n, i in enumerate(current_pubs):
                if current_entry in query_tree:
                    query_tree[current_entry].append(n + (nodes * current_entry) + 1)
                else:
                    query_tree[current_entry] = [n + (nodes * current_entry) + 1]
                query_ids[query_tree[current_entry][-1]] = ['UNCHECKED', i]
                if query_ids[query_tree[current_entry][-1]][1].citations == 'N/A':
                    query_ids[query_tree[current_entry][-1]] = ['NOT CITED', i]

            query_ids[current_entry][0] = 'CITED'

            if nodes != 1:
                graph_title = f'Root paper "{query_ids[0][1].title}", depth of {depth} with {nodes} nodes'
            else:
                graph_title = f'Root paper "{query_ids[0][1].title}", depth of {depth} with {nodes} node'

    fig = create_graph(query_tree, query_ids, graph_title)

    get_relevant_info = lambda val: [str(val), val.blurb, val.link, val.other_links]
    query_results = {str(key) : generate_paper_html(*get_relevant_info(value[1])) for key, value in query_ids.items()}
    for key, value in query_results.items():
        session[key] = value

    div = fig.to_html(full_html=False)

    return render_template('index.html', plot_div=div)

@app.route('/', methods=['GET', 'POST'])      
def index():
    if request.method == 'POST':
        article_query, depth, nodes = request.form['article'], int(request.form['depth']), int(request.form['nodes'])

        print(f'input: {article_query}, {depth = }, {nodes = }')

        pub = Paper()
        try:
            pub.get_from_query(article_query)
        except:
            return render_template('index.html', plot_div=f"<br><br><h2><center>{article_query} not found, or API inaccessible</center></h2>")

        return generate_graph_html(pub, depth, nodes)
    
    return render_template('index.html', plot_div='')

@app.route('/paper/<num>')
def paper_page(num):
    if num in session:
        paper_page_html = session[num]
    else:
        paper_page_html = '<h1>404 not found<h1>'

    return paper_page_html
  
if __name__=='__main__':
    app.run(debug=True, port=7000)
