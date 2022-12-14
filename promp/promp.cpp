#include <promp/io/serializer.hpp>
#include <promp/promp.hpp>
#include <promp/trajectory.hpp>

#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include <cassert>

// compilation : g++ ./promp.cpp  -std=c++17 -I /usr/local/include/eigen3 -lpromp
// usage : ./a.out 100 test_trajectory.csv training_trajectories*.csv
int main(int argc, char** argv)
{

    Eigen::IOFormat csv_format(Eigen::FullPrecision, 0, ",");

    // % of timesteps from the testing trajectory that we keep [0, % * len(test_traj)]
    size_t test_steps = atoi(argv[1]);

    size_t dof = atoi(argv[2]);
    std::vector<size_t> dofs = {dof}; // this is the columns that we keep

    std::cout << "Loading data...";
    std::cout.flush();

    // trajectory for conditioning (restricted to the "known" part)
    promp::Trajectory test_trajectory = promp::Trajectory(promp::io::CSVReader(argv[3]).get_data().topRows(test_steps), 1.0);
    promp::Trajectory sub_trajectory = test_trajectory.sub_trajectory(dofs);
    assert(test_steps <= sub_trajectory.timesteps());

    // vector of strings containing file names of all .csv files (each .csv file contains data from a single demonstration)
    std::vector<std::string> file_list;
    for (int i = 4; i < argc; ++i)
        file_list.push_back(std::string(argv[i]));

    // gather all the trajectories
    promp::TrajectoryGroup trajectory_group;
    trajectory_group.load_csv_trajectories(file_list, dofs);
    // normalize the lengths
    size_t t_len = trajectory_group.normalize_length();
    std::cout << "ok" << std::endl;

    /// learn promp object with number of basis functions and std as arguments.
    std::cout << "Learning promp...";
    std::cout.flush();
    static const int n_rbf = 20;
    promp::ProMP m_promp(trajectory_group, n_rbf);
    Eigen::MatrixXd promp_mean = m_promp.generate_trajectory();
    Eigen::MatrixXd promp_std = m_promp.gen_traj_std_dev();
    auto promp_cov = m_promp.generate_trajectory_covariance();
    std::cout << "ok" << std::endl;
    // write the mean promp
    {
        std::ofstream out("generated.csv");
        out << promp_mean.format(csv_format);
    }

    {
        std::ofstream out("variance.csv");
        out << promp_std.format(csv_format);
    }
    // save the *normalized* trajectories (for reference since the ProMP is learned on normalized trajectories)
    for (size_t i = 0; i < trajectory_group.trajectories().size(); ++i) {
        promp::io::save_trajectory("traj_" + std::to_string(i) + ".csv", trajectory_group.trajectories()[i]);
    }

    // modulate the PromP ?
    std::cout << "Infering speed of test traj...";
    std::cout.flush();
    double alpha = sub_trajectory.infer_speed(promp_mean, 0.75, 1.25, 2000);
    std::cout << "->" << alpha;
  
    promp::Trajectory modulated_traj = sub_trajectory.modulate(sub_trajectory.timesteps() / alpha);

      {
        std::ofstream out("modulated.csv");
        out << modulated_traj.matrix().format(csv_format);
    }

    std::cout << "Condititioning...";
    std::cout.flush();
    std::vector<std::tuple<int, Eigen::VectorXd, Eigen::MatrixXd>> via_points;
    Eigen::MatrixXd via_point_std = 1e-4 * Eigen::MatrixXd::Identity(dofs.size(), dofs.size());
    for (int i = 0; i < modulated_traj.timesteps(); ++i)
        via_points.push_back(std::make_tuple(i, modulated_traj.matrix().row(i), via_point_std));
    m_promp.condition_via_points(via_points);
    std::cout << "# POINTS:" << via_points.size() << std::endl;
    promp_mean = m_promp.generate_trajectory_with_speed(1. / alpha);
    {
        std::ofstream out("conditioned.csv");
        out << promp_mean.format(csv_format);
    }
    std::cout << "ok" << std::endl;
    return 0;
}