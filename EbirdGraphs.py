import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import numpy as np
import os
import shutil
import math


import EBirdUtil


def menu():
    print("Graphs:")


def create_im_graph(matrix, title : str, color_label : str, eps_list, teps_list):
    # Create image figure using plotly express
    fig = px.imshow(matrix.T,
                    x=eps_list,
                    y=teps_list,
                    color_continuous_scale='RdBu_r',
                    labels=dict(x="eps (miles)", y="teps (days)", color=color_label),
                    color_continuous_midpoint=0,
                    text_auto=True,
                    origin='lower')

    # Aded title and axis marks
    fig.update_layout(
        title=title,
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="RebeccaPurple"
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=np.linspace(0, int(eps_list[-1]), int(eps_list[-1]) + 1),
            ticktext=np.linspace(0, int(eps_list[-1]), int(eps_list[-1]) + 1)
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=np.linspace(0, int(teps_list[-1]), int(teps_list[-1]) + 1),
            ticktext=np.linspace(0, int(teps_list[-1]), int(teps_list[-1]) + 1)
        ),
        coloraxis_colorbar=dict(
            thicknessmode="pixels", thickness=25,
            lenmode="pixels", len=500,
        ),
        showlegend=False,
        margin=dict(l=35, r=35, t=35, b=35),
    )
    return fig


# Create pairwise comparison graphs across a range of parameter space, if two species are given, open an interactive graph in the browser,
# if more species are given, save the graphs to file
def species_pairwise_comp(species_list : list, eps_list, teps_list, save_dir="pairwise_graphs/"):
    # Clean up previously saved graphs
    for folder in os.listdir(save_dir):
        if os.path.isdir(save_dir + folder):
            shutil.rmtree(save_dir + folder)

    # Get the rankings from file, and convert to matrix form
    rank_dict = EBirdUtil.read_ranking_files()
    ranks_matrix, prams_matrix = EBirdUtil.rank_dict_to_matrix(rank_dict, len(eps_list), len(teps_list))

    # Iterate through each pair of species
    for speciesA in species_list:
        for speciesB in species_list:
            if speciesA == speciesB:
                continue

            print("Generating Pairwise Graph: {} vs {}".format(speciesA, speciesB))

            # Create comparison matrix
            comp_matrix = np.empty(ranks_matrix.shape, dtype=int)
            for i in range(len(ranks_matrix)):
                for j in range(len(ranks_matrix[i])):
                    rank: dict = ranks_matrix[i][j]

                    # Clusters of Species A - Species B, higher number meaning Species A more common, 0 means equal cluster count
                    # CURRENTLY DOES NOT SUPPORT BUILT IN TIES IN THE DICT
                    # WTF
                    comp_matrix[i][j] = next((next(count for name, count in v if name == speciesA)) for k, v in iter(rank.items()) if any(name == speciesA for (name, count) in v)) - next((next(count for name, count in v if name == speciesB)) for k, v in iter(rank.items()) if any(name == speciesB for (name, count) in v))

            # Create image figure using plotly express
            fig = create_im_graph(comp_matrix, '<span style="font-size: 24px;">{0} vs {1} Rarity</span><br><sup>Calculated as # of Clusters of {0} - {1} (Blue means {0} is more rare)</sup>'.format(speciesA, speciesB), "Cluster Count Difference", eps_list, teps_list)

            # If two species are provided, open interactive graph, else save graph to save folder
            if len(species_list) == 2:
                fig.open()
            else:
                if not os.path.isdir(save_dir + "{}".format(speciesA)):
                    os.mkdir(save_dir + "{}".format(speciesA))
                fig.write_image(save_dir + "{}/{} vs {}.png".format(speciesA, speciesA, speciesB), width=1200, height=800, scale=2)


def max_localized_difference_graph(radius, eps_list, teps_list):
    # Get the rankings from file, and convert to matrix form
    rank_dict = EBirdUtil.read_ranking_files()
    ranks_matrix, prams_matrix = EBirdUtil.rank_dict_to_matrix(rank_dict, len(eps_list), len(teps_list), flatten=True)
    # print(ranks_matrix)
    diff_matrix = np.empty(ranks_matrix.shape, dtype=float)
    for i in range(len(ranks_matrix)):
        for j in range(len(ranks_matrix[i])):
            max_diff = 0
            for i1 in range(len(ranks_matrix)):
                for j1 in range(len(ranks_matrix[i1])):
                    if math.dist([eps_list[i], teps_list[j]], [eps_list[i1], teps_list[j1]]) < radius:
                        diff = EBirdUtil.normalized_kendall_tau_distance(ranks_matrix[i][j], ranks_matrix[i1][j1])
                        max_diff = diff if diff > max_diff else max_diff

            diff_matrix[i][j] = max_diff

    # Create image figure using plotly express
    fig = px.imshow(diff_matrix.T,
                    x=eps_list,
                    y=teps_list,
                    color_continuous_scale='RdBu_r',
                    labels=dict(x="eps (miles)", y="teps (days)", color="Normalized kendall tau distance"),
                    color_continuous_midpoint=0,
                    text_auto=True,
                    origin='lower')

    # Aded title and axis marks
    fig.update_layout(
        title='<span style="font-size: 24px;">Maximum kendall tau distance score in radius {}</span>'.format(radius),
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="RebeccaPurple"
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=np.linspace(0, int(eps_list[-1]), int(eps_list[-1]) + 1),
            ticktext=np.linspace(0, int(eps_list[-1]), int(eps_list[-1]) + 1)
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=np.linspace(0, int(teps_list[-1]), int(teps_list[-1]) + 1),
            ticktext=np.linspace(0, int(teps_list[-1]), int(teps_list[-1]) + 1)
        ),
        coloraxis_colorbar=dict(
            thicknessmode="pixels", thickness=25,
            lenmode="pixels", len=500,
        ),
        showlegend=False,
        margin=dict(l=35, r=35, t=35, b=35),
    )

    fig.show()


if __name__ == "__main__":
    old_species = [
        "Hudsonian Godwit", "Ruff", "Groove-billed Ani", "Acorn Woodpecker", "Brown Thrasher", "Eastern Phoebe", "Gray Catbird", "Huttons Vireo", "Lark Bunting", "Lesser Black-backed Gull", "Long-tailed Duck", "Long-tailed Jaeger", "Mew Gull", "Parasitic Jaeger", "Pomarine Jaeger", "Red Phalarope", "Red-faced Warbler", "Sabines Gull"
    ]
    fall_species = [
        "Black-and-white Warbler", "Black-throated Blue Warbler", "Common Tern", "Groove-billed Ani", "Kentucky Warbler", "Lesser Black-backed Gull", "Long-tailed Jaeger", "Nelsons Sparrow", "Painted Bunting", "Parasitic Jaeger", "Prothonotary Warbler", "Roseate Spoonbill", "Ruff", "Rufous-backed Robin", "Sabines Gull", "Tennessee Warbler", "White Ibis"
    ]
    winter_species = [
        "Blue Jay", "Bohemian Waxwing", "Bonapartes Gull", "Brown Thrasher", "Common Redpoll", "Couchs Kingbird", "Glaucous Gull", "Groove-billed Ani", "Huttons Vireo", "Lark Bunting", "Lesser Black-backed Gull", "Long-tailed Duck", "Red-necked Grebe", "Red-throated Loon", "Short-billed Gull", "Slaty-backed Gull"
    ]
    # species = ["Hudsonian Godwit", "Pomarine Jaeger"]
    # menu()
    EBirdUtil.generate_ranking_files(winter_species, np.linspace(.5, 30, 60), np.linspace(.5, 30, 60))
    species_pairwise_comp(winter_species, np.linspace(.5, 30, 60), np.linspace(.5, 30, 60))
    # max_localized_difference_graph(5, np.linspace(.5, 30, 60), np.linspace(.5, 30, 60))