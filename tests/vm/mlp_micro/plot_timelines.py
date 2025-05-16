import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pickle
import sys

# Set global font sizes
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

all_timings_1stage = pickle.load(open('timings_1stage.pkl', 'rb'))
all_timings_2stage = pickle.load(open('timings_2stage.pkl', 'rb'))

print()

print(all_timings_1stage[:128,1,60].argmax())
print(all_timings_2stage[:128,1,60].argmin())

print()

timings_1stage_op1, timings_1stage_op2 = all_timings_1stage[int(sys.argv[1]),0], all_timings_1stage[int(sys.argv[1]),1]
timings_2stage_op1, timings_2stage_op2 = all_timings_2stage[int(sys.argv[2]),0], all_timings_2stage[int(sys.argv[2]),1]

print(timings_1stage_op1[[0,1,10,30,45,50,60]])
print(timings_1stage_op2[[0,1,10,30,45,50,60]])

timings_1stage_op1 = timings_1stage_op1 / 1e3
timings_1stage_op2 = timings_1stage_op2 / 1e3

timings_2stage_op1 = timings_2stage_op1 / 1e3
timings_2stage_op2 = timings_2stage_op2 / 1e3

print(timings_1stage_op1)
print(timings_1stage_op2)

def get_events(timings_op1, timings_op2):
    if timings_op1[30] > timings_op2[10]:
        mid = (timings_op2[10] + timings_op1[30]) / 2
        timings_op1[30], timings_op2[10] = mid, mid
    active_times = {
        'Controller': [(timings_op1[0], timings_op1[1]), (timings_op2[0], timings_op2[1])],
        'Loader': [(timings_op1[10], timings_op1[30]), (timings_op2[10], timings_op2[30])],
        'Consumer': [(timings_op1[30], timings_op1[45]), (timings_op2[30], timings_op2[45])],
        'Storer': [(timings_op1[50], timings_op1[60]), (timings_op2[50], timings_op2[60])]
    }

    return active_times, max(timings_op1.max(), timings_op2.max())

active_times_1stage, max_time_1stage = get_events(timings_1stage_op1, timings_1stage_op2)
active_times_2stage, max_time_2stage = get_events(timings_2stage_op1, timings_2stage_op2)

max_time = max(max_time_1stage, max_time_2stage)

# Define Tableau 10 color palette
tableau10_colors = ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2', 
                   '#59a14f', '#edc949', '#af7aa1', '#ff9da7', 
                   '#9c755f', '#bab0ab']

# Assign Tableau 10 colors to each task category
task_colors = {
    'Controller': tableau10_colors[0],  # Blue
    'Loader': tableau10_colors[1],      # Orange
    'Consumer': tableau10_colors[2],    # Red
    'Storer': tableau10_colors[3]       # Teal
}

def plot_gantt_chart(ax, active_times, max_time, title, divider = "straight", time_label = True, use_y_labels = True):
    """Plots a Gantt chart on the given axes."""
    y_labels = list(active_times.keys())
    y_labels.reverse()
    y_ticks = range(len(y_labels))

    red_line_connection_points = [[max_time+0.5], [-0.5]]

    # Calculate connection points for the jagged divider line
    for i, task_name in enumerate(y_labels):
        task_intervals = active_times[task_name] # This is [(s1, e1), (s2, e2)]
        red_line_connection_points[0].append((task_intervals[0][1] + task_intervals[1][0]) / 2)
        red_line_connection_points[1].append(i - 0.5)

        red_line_connection_points[0].append((task_intervals[0][1] + task_intervals[1][0]) / 2)
        red_line_connection_points[1].append(i + 0.5)

    red_line_connection_points[0].append(0)
    red_line_connection_points[1].append(i + 0.5)

    # Draw divider lines first (in the background)
    if divider == "straight":
        b1 = max(active_times[task_name][0][1] for task_name in active_times)
        b2 = min(active_times[task_name][1][0] for task_name in active_times)
        mid = (b1 + b2) / 2
        ax.axvline(mid, color='red', linestyle='--', zorder=0)
    else:
        ax.plot(red_line_connection_points[0], red_line_connection_points[1], color='red', linestyle='--', zorder=0)
    
    # Draw bars on top
    for i, task_name in enumerate(y_labels):
        task_intervals = active_times[task_name] # This is [(s1, e1), (s2, e2)]
        
        # Plot the two bars for this task with consistent color and black edge
        ax.barh(i, task_intervals[0][1] - task_intervals[0][0], left=task_intervals[0][0], 
                height=0.7, align='center', color=task_colors[task_name], 
                edgecolor='black', linewidth=0.5, zorder=1)
        ax.barh(i, task_intervals[1][1] - task_intervals[1][0], left=task_intervals[1][0], 
                height=0.7, align='center', color=task_colors[task_name], 
                edgecolor='black', linewidth=0.5, zorder=1)
        
    span = (-0.5, 0.5)
    ax.axhspan(span[0], span[1], color='lightgrey', alpha=0.2)
    ax.axhspan(2 + span[0], 2 + span[1], color='lightgrey', alpha=0.2)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    
    if time_label:
        ax.set_xlabel("Time (µs)")
    
    if use_y_labels:
        ax.set_yticklabels(y_labels)
    else:
        print("hiding y labels")
        ax.set_yticks([])
    
    ax.set_title(title)
    ax.set_xlim(0, max_time+0.5)
    ax.set_ylim(-0.6, len(y_labels) - 0.4)
    ax.grid(True, axis='x')

    # # Add a legend for task colors
    # legend_patches = [mpatches.Patch(color=color, label=task, edgecolor='black', linewidth=0.5) 
    #                 for task, color in task_colors.items()]
    # ax.legend(handles=legend_patches, loc='lower left')


if __name__ == "__main__":

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4), sharex=True)

    # Plot Gantt chart for 1-stage pipeline
    plot_gantt_chart(ax1, active_times_1stage, max_time, "1-Stage Pipeline", divider="straight")

    # Plot Gantt chart for 2-stage pipeline
    plot_gantt_chart(ax2, active_times_2stage, max_time, "2-Stage Pipeline", divider="jagged", use_y_labels=False)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig('gantt_chart.png')