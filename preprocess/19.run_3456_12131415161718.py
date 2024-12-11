import os
import warnings
from config import parser


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    args = parser.parse_args()

    os.system("python 03.data_filter.py")
    print("finish 03.data_filter.py")

    os.system("python 04.demand_data_selection.py")
    print("finish 04.deman_data_selection.py")

    os.system("python 05.time_format_change.py")
    print("finish 05.time_format_change.py")

    os.system("python 06.demand_data_filter.py")
    print("finish 06.demand_data_filter.py")

    os.system("python 12.sumo_personDemandtrips_generate.py")
    print("finish 12.sumo_personDemandtrips_generate.py")

    os.system("python 13.demand_pruning.py")
    print("finish 13.demand_pruning.py")

    os.system("python 14.demand_graph.py")
    print("finish 14.demand_graph.py")

    os.system("python 15.data_sample.py")
    print("finish 15.data_sample.py")

    os.system("python 16.sample_trajectory.py")
    print("finish 16.sample_trajectory.py")

    os.system("python 17.sample_graph.py")
    print("finish 17.sample_graph.py")

    os.system("python 18.one_car_trajectory_generate.py")
    print("finish 18.one_car_trajectory_generate.py")

    print("finsh" + args.table[-4:] + "!!!")