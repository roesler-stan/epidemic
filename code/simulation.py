"""
    Simulate the coronavirus pandemic

    python3 simulation.py runs these simulations and saves results into ../output

    The current setup takes ~4 hours.
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import os
import shutil
import math
import imageio
from PIL import Image
import pygifsicle
from constants import *


# Constants.  You could alter this file to vary these, e.g. to try different P(infect) parameters
N = 1000
N_CLUSTERS = 10
N_ITERATIONS = 100
N_DAYS = 364
N_INITIAL_INFECTIONS = 2
# Number of nodes that are social
N_SOCIAL = 10
# P(infect) = 0.0538 -> R0 = 1 (given E(infected days | infected) = 18.6)
P_INFECT =  0.1
P_DIE = 0.001
P_RECOVER = 0.05
# How many days after initial infection to use for R0 calculation
R0_DAYS = 30


class Simulation():
    """
        A Simulation object, to be run with given parameters

        Within the simulation, several values for P(infect) are used
    """
    def __init__(self, version, non_social_within_cluster_contacts, social_within_cluster_contacts, cross_cluster_contacts):
        self.version = version
        # Directory to put all output into for this version
        self.output_dir = f'../output/simulations/v{version}'
        # How many infections are present on day 0
        self.n_initial_infections = N_INITIAL_INFECTIONS
        self.p_infect = P_INFECT
        self.n_social = N_SOCIAL
        self.non_social_within_cluster_contacts = non_social_within_cluster_contacts
        self.social_within_cluster_contacts = social_within_cluster_contacts
        self.cross_cluster_contacts = cross_cluster_contacts

        # 10k takes a few minutes to write in Python and is too slow in Gephi
        self.N = N
        # % of population that is social (e.g. 0.25% doctors, some supermarket workers)
        # Clusters (like countries or big cities within the US)
        self.n_clusters = N_CLUSTERS
        # How many days from initial infections to forecast
        self.n_days = N_DAYS
        self.R0_days = R0_DAYS
        assert self.n_days >= self.R0_days
        # How many simulations to run per version
        self.n_iterations = N_ITERATIONS
        # Daily probability of dying or recovering, given infection
        self.p_die = P_DIE
        self.p_recover = P_RECOVER

    def _prepare_directory(self):
        os.makedirs(self.output_dir)
        os.makedirs(f"{self.output_dir}/plots")
        os.makedirs(f"{self.output_dir}/data")
        os.makedirs(f"{self.output_dir}/graphs")

    def _store_version_info(self):
        """Store constants for this version."""
        df = pd.DataFrame.from_dict({
            'version': self.version,
            'N': self.N,
            'n_social': self.n_social,
            'n_clusters': self.n_clusters,
            'n_days': self.n_days,
            'n_iterations': self.n_iterations,
            'n_initial_infections': self.n_initial_infections,
            'p_infect': self.p_infect,
            'p_die': self.p_die,
            'p_recover': self.p_recover,
            'non_social_within_cluster_contacts': self.non_social_within_cluster_contacts,
            'social_within_cluster_contacts': self.social_within_cluster_contacts,
            'cross_cluster_contacts': self.cross_cluster_contacts,
        }, orient='index')
        filename = f'{self.output_dir}/version_info.csv'
        df.T.to_csv(filename, index=False)

    def _create_graph(self):
        """Create a new graph, without any infections.

        The simulation can have only a single graph at a time.
        """
        self.G = nx.DiGraph()

        ## Create each node
        self.G.add_nodes_from(range(self.N))
        for node in self.G.nodes:
            self.G.nodes[node]['social'] = False
            self.G.nodes[node]['cluster'] = node % self.n_clusters
            self.G.nodes[node]['state'] = 'susceptible'

        # Randomly select social nodes
        for node in random.sample(range(0, self.G.number_of_nodes()-1), self.n_social):
            self.G.nodes[node]['social'] = True

        ## Draw edges
        # Deal with fact that cross-cluster contacts can be fractional
        cross_cluster_contacts_int = math.ceil(self.cross_cluster_contacts)
        if cross_cluster_contacts_int == 0:
            cross_cluster_contacts_proportion = 0
        else:
            cross_cluster_contacts_proportion = self.cross_cluster_contacts / cross_cluster_contacts_int

        # Go through each cluster
        for cluster in range(self.n_clusters):
            this_cluster_nodes = [node for node in self.G.nodes if self.G.nodes[node]['cluster'] == cluster]
            other_cluster_nodes = [node for node in self.G.nodes if self.G.nodes[node]['cluster'] != cluster]
            for node_i in this_cluster_nodes:
                # Within-cluster edges
                if self.G.nodes[node_i]['social']:
                    n_allowed_edges = self.social_within_cluster_contacts
                else:
                    n_allowed_edges = self.non_social_within_cluster_contacts
                valid_other_nodes = [node for node in this_cluster_nodes if node != node_i]
                for node_j in random.sample(valid_other_nodes, n_allowed_edges):
                    self.G.add_edge(node_i, node_j)

                # Cross-cluster edges
                valid_other_nodes = [node for node in other_cluster_nodes]
                for node_j in random.sample(valid_other_nodes, cross_cluster_contacts_int):
                    self.G.add_edge(node_i, node_j)

        # Remove excessive cross-cluster ties to get desired fractional value
        # Create a copy b/c the edges will change as you remove them
        for edge in self.G.copy().edges:
            node_i = edge[0]
            node_j = edge[1]
            if self.G.nodes[node_i]['cluster'] != self.G.nodes[node_j]['cluster'] and random.random() < (1 - cross_cluster_contacts_proportion):
                self.G.remove_edge(node_i, node_j)

        # Make sure that intended degree is roughly what is produced
        # Technically, within-cluster degree should be precise, while cross-cluster is probabalistic
        self.social_degree_actual = np.mean([v for k, v in self.G.out_degree() if self.G.nodes[k]['social']])
        self.social_degree_intended = (self.social_within_cluster_contacts + self.cross_cluster_contacts)
        social_degree_ratio = self.social_degree_actual / self.social_degree_intended
        assert social_degree_ratio > 0.95 and social_degree_ratio < 1.05, "Social nodes do not have intended # of contacts"

        self.non_social_degree_actual = np.mean([v for k, v in self.G.out_degree() if not self.G.nodes[k]['social']])
        self.non_social_degree_intended = (self.non_social_within_cluster_contacts + self.cross_cluster_contacts)
        non_social_degree_ratio = self.non_social_degree_actual / self.non_social_degree_intended
        assert non_social_degree_ratio > 0.95 and non_social_degree_ratio < 1.05, "Non-social nodes do not have intended # of contacts"

        max_pct_edgeless_nodes = 0.1
        nodes_without_edges = len([k for k, v in self.G.out_degree() if v == 0])
        assert (nodes_without_edges / self.N) * 100 <= max_pct_edgeless_nodes, f"> {max_pct_edgeless_nodes}% nodes have no edges"

        # Select a single layout for this instance of the graph
        self.graph_layout = nx.spring_layout(self.G)

    def _draw_network(self, day, iteration):
        fig, ax = plt.subplots()
        for state in ['susceptible', 'infected', 'recovered', 'dead']:
            # Draw nodes
            state_nodes = [node for node in self.G.nodes if self.G.nodes[node]['state'] == state]
            if not state_nodes:
                continue
            node_degrees = [val for (node, val) in self.G.degree() if node in state_nodes]
            # Normalize degree to be between 0 and 100
            if np.std(node_degrees) == 0:
                node_degrees_normalized = [50 for node in node_degrees]
            else:
                node_degrees_normalized = (([n for n in node_degrees] - np.min(node_degrees)) / (np.max(node_degrees) - np.min(node_degrees))) * 100
            nx.draw_networkx_nodes(
                self.G,
                self.graph_layout,
                nodelist=state_nodes,
                node_color=state_colors[state],
                node_size=node_degrees_normalized,
                alpha=0.8,
            )

        # Draw edges
        nx.draw_networkx_edges(
            self.G,
            self.graph_layout,
            edge_color='black',
            width=0.5,
            alpha=0.5,
        )
        outfile = f'{self.output_dir}/plots/iteration{iteration}_day{day}.png'
        fig.savefig(outfile)
        plt.close()

    def _run_time(self, iteration, p_infect):
        """ Step through time to see how disease progresses

        Day 0: one person infected
        Day 1: first time the virus can spread person-to-person
        ...

        """

        df_iteration = pd.DataFrame()
        for day in range(self.n_days + 1):
            new_infections = 0
            if day == 0:
                for node in random.sample(range(0, self.G.number_of_nodes()-1), self.n_initial_infections):
                    self.G.nodes[node]['state'] = 'infected'
                    self.G.nodes[node]['infection_day'] = day
                    self.G.nodes[node]['n_ever_infected'] = 0
                    new_infections += 1
            else:
                # Some nodes die
                for node in self.G.nodes:
                    if self.G.nodes[node]['state'] == 'infected' and random.random() < self.p_die:
                        self.G.nodes[node]['state'] = 'dead'
                # Some nodes recover and are no longer susceptible
                for node in self.G.nodes:
                    if self.G.nodes[node]['state'] == 'infected' and random.random() < self.p_recover:
                        self.G.nodes[node]['state'] = 'recovered'
                # Some nodes infect others
                for edge in [e for e in self.G.edges]:
                    node_i = edge[0]
                    node_j = edge[1]
                    if self.G.nodes[node_i]['state'] == 'infected' and self.G.nodes[node_j]['state'] == 'susceptible' and random.random() < p_infect:
                        self.G.nodes[node_j]['state'] = 'infected'
                        self.G.nodes[node_j]['infection_day'] = day
                        # Keep track of how many nodes have infected others
                        if self.G.nodes[node_i]['infection_day'] <= day - self.R0_days:
                            self.G.nodes[node_i]['n_ever_infected'] += 1
                        self.G.nodes[node_j]['n_ever_infected'] = 0
                        new_infections += 1

            # Save a snapshot of the graph every 10 days and every day in the beginning
            # if iteration == 0 and (day < 5 or day % 10 == 0):
            if iteration == 0:
                # Draw using NetworkX
                self._draw_network(day, iteration)
                # Save to Gephi
                outfile = f'{self.output_dir}/graphs/day{day}_iteration{iteration}.gexf'
                nx.write_gexf(self.G, outfile)

            states = list(nx.get_node_attributes(self.G, 'state').values())
            n_infected = states.count('infected')
            n_susceptible = states.count('susceptible')
            n_recovered = states.count('recovered')
            n_dead = states.count('dead')

            ## Save values for this day
            row = {
                'iteration': iteration,
                'day': day,
                'social_degree_actual': self.social_degree_actual,
                'non_social_degree_actual': self.non_social_degree_actual,
                'social_degree_intended': self.social_degree_intended,
                'non_social_degree_intended': self.non_social_degree_intended,
                'susceptible': n_susceptible,
                'infected': n_infected,
                'recovered': n_recovered,
                'dead': n_dead,
                'new_infections': new_infections,
            }
            df_iteration = df_iteration.append(row, ignore_index=True)

        # Calculate R0: how many other people each person ever infects
        # We want R0 for each day, where day is the day the first person was infected
        # and R0 is how many people she eventually infected (at most 30 days after her initial infection)
        # Censor it for people who are infected <30 days before the end of the observed period
        df_R0_iteration = pd.DataFrame()
        for infection_day in range(self.n_days - self.R0_days + 1):
            n_infected = []
            for node in self.G.nodes:
                node_info = self.G.nodes[node]
                if 'infection_day' in node_info and node_info['infection_day'] == infection_day:
                    n_infected.append(node_info['n_ever_infected'])
            row = {
                'day': infection_day,
                'iteration': iteration,
                'R0': round(np.mean(n_infected), 2),
            }
            df_R0_iteration = df_R0_iteration.append(row, ignore_index=True)

        self.df = self.df.append(df_iteration, ignore_index=True)
        self.df_R0 = self.df_R0.append(df_R0_iteration, ignore_index=True)

    def _run_simulation(self):
        self.df = pd.DataFrame()
        self.df_R0 = pd.DataFrame()
        # For each iteration, create a new graph and run it through time
        for iteration in range(self.n_iterations):
            self._create_graph()
            self._run_time(iteration, self.p_infect)
        # Merge df with R0 data now that both are complete
        # R0 data is right-censored, so don't require it to be present
        self.df = self.df.merge(self.df_R0, on=['day', 'iteration'], how='left')

    def _save_df(self):
        outfile = f'{self.output_dir}/data/final.csv'
        self.df.to_csv(outfile, index=False)

    def run_and_save(self):
        self._prepare_directory()
        self._store_version_info()
        self._run_simulation()
        self._save_df()


def _run_simulations():
    # Delete all old files
    if os.path.isdir("../output"):
        shutil.rmtree("../output")
    os.makedirs("../output/simulations")

    version = 0
    for non_social_within_cluster_contacts in [2, 4]:
        for cross_cluster_contacts in [0.1, 0.2]:
            # Social people have more within-cluster contacts
            social_within_cluster_contacts = non_social_within_cluster_contacts * 5

            simulation = Simulation(
                version=version,
                non_social_within_cluster_contacts=non_social_within_cluster_contacts,
                social_within_cluster_contacts=social_within_cluster_contacts,
                cross_cluster_contacts=cross_cluster_contacts,
            )
            simulation.run_and_save()
            version += 1


def _count_versions():
    """Read how many simluations were run."""
    for (dirpath, dirnames, filenames) in os.walk("../output/simulations"):
        return len(dirnames)


def _save_meta_df():
    """
        Read files in output directory and make final DataFrames
    """
    output_dir = "../output/data"
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    version_info_df = pd.DataFrame()
    raw_df = pd.DataFrame()

    n_versions = _count_versions()
    for version in range(n_versions):
        directory = f"../output/simulations/v{version}"
        # Add to version info DF
        version_info = pd.read_csv(f"{directory}/version_info.csv")
        version_info_df = version_info_df.append(version_info, ignore_index=True)
        # Add to overall raw DF (version-iteration-day level)
        version_df = pd.read_csv(f"{directory}/data/final.csv")
        version_df['version'] = version
        raw_df = raw_df.append(version_df, ignore_index=True)

    # Summary DF
    cols = ['susceptible', 'infected', 'recovered', 'dead', 'new_infections', 'R0', 'non_social_degree_actual', 'social_degree_actual']
    means = raw_df.groupby(['version', 'day'])[cols].mean().round(2)
    n_iterations = raw_df['iteration'].max() + 1
    se = (raw_df.groupby(['version', 'day'])[cols].mean() / np.sqrt(n_iterations)).round(2)
    means.rename(columns=lambda x: x + '_mean', inplace=True)
    se.rename(columns=lambda x: x + '_se', inplace=True)
    means.reset_index(inplace=True)
    se.reset_index(inplace=True)
    summary_df = means.merge(se, on=['version', 'day'])
    summary_df = summary_df.merge(version_info_df, on=['version'])

    version_info_df.to_csv(f"{output_dir}/version_info.csv", index=False)
    raw_df.to_csv(f"{output_dir}/raw.csv", index=False)
    summary_df.to_csv(f"{output_dir}/summary.csv", index=False)


def _make_meta_tables():
    output_dir = "../output/tables"
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    df = pd.read_csv("../output/data/summary.csv")

    # 2x2 table of within- and cross-cluster contacts and CI of states on last day
    for outcome in ['susceptible', 'infected', 'recovered', 'dead']:
        # Convert to %
        df[f'{outcome}_mean'] = (df[f'{outcome}_mean'] / N) * 100
        df[f'{outcome}_se'] = (df[f'{outcome}_se'] / N) * 100
        df[f'{outcome}_ci'] = (df[f"{outcome}_mean"] - (1.96 * df[f"{outcome}_se"])).round(1).astype(str) + '-' + \
        (df[f"{outcome}_mean"] + (1.96 * df[f"{outcome}_se"])).round(1).astype(str)
        table = df[df['day'] == N_DAYS].pivot(index='non_social_within_cluster_contacts', columns='cross_cluster_contacts', values=f"{outcome}_ci")
        table.to_csv(f"{output_dir}/{outcome}_day{N_DAYS}.csv", index=True)

    for col in ['social_degree_actual_mean', 'non_social_degree_actual_mean']:
        table = df.loc[df['day'] == 0, ['version', col]]
        table.to_csv(f"{output_dir}/{col}.csv", index=False)


def _make_meta_plots():
    output_dir = "../output/plots"
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    df = pd.read_csv("../output/data/summary.csv")

    # States by day - within each version
    for version in df['version'].unique():
        subset = df[df['version'] == version]
        non_social_within_cluster_contacts = int(subset['non_social_within_cluster_contacts'].mean())
        cross_cluster_contacts = round(subset['cross_cluster_contacts'].mean(), 1)

        fig, ax = plt.subplots()
        for state in ['susceptible', 'infected', 'recovered', 'dead']:
            # It's already sorted by day
            plt.errorbar(
                x=subset['day'],
                y=(subset[f'{state}_mean'] / N) * 100,  # Convert to %
                yerr=((subset[f'{state}_se'] * 1.96) / N) * 100,
                label=state,
                color=state_colors[state],
            )

        plt.ylim(0, 100)
        plt.xlabel(f'Days since initial {N_INITIAL_INFECTIONS} infections')
        plt.ylabel('% of population')
        plt.title(f"{non_social_within_cluster_contacts} non-social within-cluster contacts,\n{cross_cluster_contacts} cross-cluster contacts")
        plt.subplots_adjust(left=0.16, right=0.79)
        plt.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize=12,
            frameon=True,
        ).set_title('State', prop={'size': 'xx-small'})

        outfile = f"{output_dir}/states_v{version}.png"
        fig.savefig(outfile)
        plt.close()


def _make_meta_gifs():
    output_dir = "../output/gifs"
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)    

    n_versions = _count_versions()
    for version in range(n_versions):
        directory = f"../output/simulations/v{version}/plots"
        images = []
        for day in range(N_DAYS + 1):
            filename = os.path.join(directory, f'iteration0_day{day}.png')
            # Crop image, original is 1100 x 800
            left = 130
            right = 1000
            top = 90
            bottom = 720
            im = Image.open(filename)
            im_cropped = im.crop((left, top, right, bottom))
            images.append(im_cropped)

        outfile = os.path.join(output_dir, f"v{version}.gif")
        imageio.mimsave(outfile, images)
        # Optimize to reduce storage
        pygifsicle.optimize(outfile)


if __name__ == "__main__":
    _run_simulations()
    _save_meta_df()
    _make_meta_tables()
    _make_meta_plots()
    _make_meta_gifs()
    # In shell, convert gifs to videos
    # ffmpeg -i ../output/gifs/v0.gif ../output/gifs/v0.mp4
