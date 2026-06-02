// make_training_data.cpp
//
// C++ port of make_training_data_truthmatch.py.
// Reads GEMC HIPO files and writes the compact-format .dat files used by the
// fast-MC training pipeline.  Matches MC::Particle to REC::Particle via the
// MC::RecMatch hit-based truth bank (quality > threshold).
//
// Produces 4 output files (same as the Python version):
//   <out>_train.dat               full event set, train split
//   <out>_val.dat                 full event set, val   split
//   <out>_hadgated_train.dat      subset where electron is truth-matched in FD
//   <out>_hadgated_val.dat
//
// Build with the bundled CMakeLists.txt (links against HIPO4).
//
// Usage:
//   ./make_training_data <hipo_dir> <output_dir> \
//        --reaction epK+K- --beam_energy 10.6 \
//        --beam_pid 11 --target_pid 2212 \
//        --min_electron_theta 6.0 \
//        --quality 0.98 --val_fraction 0.20 \
//        [--max_files N] [--file_offset N] [--basename phi_tm] \
//        [--seed 42]
//
//----------------------------------------------------------------------------

#include "hipo4/reader.h"

#include <algorithm>
#include <cmath>
#include <csetjmp>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// ─── Constants ────────────────────────────────────────────────────────────
constexpr double FD_CD_BOUNDARY = 35.0;           // deg
constexpr int    FD_STATUS_CUT  = 2000;
constexpr int    CD_STATUS_CUT  = 4000;
constexpr int    FT_STATUS_CUT  = 1000;
constexpr double DEG_PER_RAD    = 180.0 / M_PI;

// ─── Recovery from HIPO C++ aborts (vector assertions on bad files) ───────
// When std::vector::operator[] fires its bounds-check assertion the HIPO
// library calls abort().  We catch the resulting SIGABRT and siglongjmp back
// to the file loop, skipping the offending file and continuing.
//
// This is technically UB (POSIX forbids siglongjmp from a signal handler that
// fires during library code) — in practice on glibc + GCC it works for our
// purpose because we only touch primitive state in the handler and the HIPO
// objects are stack-scoped per file (they leak memory but the program
// continues).
static sigjmp_buf g_abort_jmp;
static volatile sig_atomic_t g_in_file = 0;

static void abort_signal_handler(int /*sig*/)
{
    if (g_in_file) {
        g_in_file = 0;
        siglongjmp(g_abort_jmp, 1);
    }
    // Outside a file — let default handler run (true crash)
    std::signal(SIGABRT, SIG_DFL);
    std::raise(SIGABRT);
}

// ─── Parsed CLI parameters ────────────────────────────────────────────────
struct Params {
    std::string hipo_dir;
    std::string output_dir;
    std::string reaction        = "epK+K-";
    std::string basename        = "phi_tm";
    std::string gated_suffix    = "hadgated";
    std::string electron_match  = "pidonly";        // pidonly|truth
    double      beam_energy     = 10.6;             // GeV
    int         beam_pid        = 11;               // electron
    int         target_pid      = 2212;             // proton
    double      target_mass     = 0.938272;
    double      min_electron_theta = 6.0;           // deg
    double      quality_thresh  = 0.98;
    double      val_fraction    = 0.20;
    int         max_files       = 0;                // 0 = all
    int         file_offset     = 0;
    unsigned    seed            = 42;
    bool        verbose         = false;
};

// ─── Reaction → expected MC PIDs (sorted, for channel matching) ───────────
static std::vector<int> expected_pids(const std::string& reaction)
{
    if (reaction == "epK+K-")    return {-321, 11, 321, 2212};
    if (reaction == "eppi+pi-")  return {-211, 11, 211, 2212};
    if (reaction == "epe+e-")    return {-11,  11, 2212};
    std::cerr << "Unknown reaction: " << reaction << std::endl;
    std::exit(1);
}

// ─── Reaction → canonical output ordering (matches Python) ───────────────
// [e-, p, h+, h-] where h+h- are the meson daughters.
// M(hh) is computed from indices 2 and 3 in this order.
static std::vector<int> canonical_pid_order(const std::string& reaction)
{
    if (reaction == "epK+K-")    return {11, 2212, -321, 321};
    if (reaction == "eppi+pi-")  return {11, 2212, -211, 211};
    if (reaction == "epe+e-")    return {11, 2212, -11};
    std::cerr << "Unknown reaction: " << reaction << std::endl;
    std::exit(1);
}

// ─── pid → short name (for log/print) ─────────────────────────────────────
static std::string pid_short(int pid) {
    switch (pid) {
        case   11: return "e-";
        case  -11: return "e+";
        case 2212: return "p";
        case 2112: return "n";
        case  211: return "pi+";
        case -211: return "pi-";
        case  321: return "K+";
        case -321: return "K-";
        case   22: return "gamma";
        default:   return std::to_string(pid);
    }
}

// ─── Detector tag from REC::Particle status word ──────────────────────────
static int det_from_status(int status)
{
    const int s = std::abs(status);
    if (s >= CD_STATUS_CUT) return 2;   // CD
    if (s >= FD_STATUS_CUT) return 1;   // FD
    if (s >= FT_STATUS_CUT) return 3;   // FT
    return 1;                            // default to FD if low/odd
}

// ─── Kinematics from (px, py, pz) ─────────────────────────────────────────
struct Kin { double p, theta, phi; };
static inline Kin kinematics(double px, double py, double pz)
{
    Kin k;
    k.p = std::sqrt(px*px + py*py + pz*pz);
    if (k.p > 0) {
        k.theta = std::acos(pz / k.p) * DEG_PER_RAD;
        k.phi   = std::atan2(py, px)  * DEG_PER_RAD;
    } else {
        k.theta = 0;  k.phi = 0;
    }
    return k;
}

// ─── Q²/xB/W/t for the (e,e') part of the event (event-line metadata) ────
struct EvtKin { double Q2, xB, W, t; };
static EvtKin compute_evt_kin(double beam_E,
                              double target_mass,
                              const Kin& e_in,
                              const Kin& e_out,
                              const Kin& proton_in,
                              const Kin& proton_out)
{
    // Lorentz inputs (we'll do 4-vectors by hand to keep dependencies minimal)
    auto en = [](double p, double m){ return std::sqrt(p*p + m*m); };
    double mE = 0.000511;
    double mP = target_mass;

    double Ein_E = beam_E;
    double Ein_pz = std::sqrt(beam_E*beam_E - mE*mE);

    double Et_E = mP;

    double Eout_E  = en(e_out.p, mE);
    double Eout_px = e_out.p * std::sin(e_out.theta/DEG_PER_RAD)*std::cos(e_out.phi/DEG_PER_RAD);
    double Eout_py = e_out.p * std::sin(e_out.theta/DEG_PER_RAD)*std::sin(e_out.phi/DEG_PER_RAD);
    double Eout_pz = e_out.p * std::cos(e_out.theta/DEG_PER_RAD);

    double Pout_E  = en(proton_out.p, mP);
    double Pout_px = proton_out.p * std::sin(proton_out.theta/DEG_PER_RAD)*std::cos(proton_out.phi/DEG_PER_RAD);
    double Pout_py = proton_out.p * std::sin(proton_out.theta/DEG_PER_RAD)*std::sin(proton_out.phi/DEG_PER_RAD);
    double Pout_pz = proton_out.p * std::cos(proton_out.theta/DEG_PER_RAD);

    // q = ein - eout
    double q_E  = Ein_E  - Eout_E;
    double q_px = 0      - Eout_px;
    double q_py = 0      - Eout_py;
    double q_pz = Ein_pz - Eout_pz;
    double Q2   = -(q_E*q_E - (q_px*q_px + q_py*q_py + q_pz*q_pz));

    // xB = Q²/(2 P·q)
    double Pq = mP * q_E;
    double xB = (Pq > 0) ? Q2 / (2 * Pq) : -999.0;

    // W² = (q + P_target)²
    double W2 = (q_E + Et_E)*(q_E + Et_E) - (q_px*q_px + q_py*q_py + q_pz*q_pz);
    double W  = (W2 > 0) ? std::sqrt(W2) : -999.0;

    // Mandelstam t = (P_recoil − P_target)² (negative for physical kinematics).
    // Python uses dE = E_recoil − M_target, dp = p_recoil; t = dE² − dp².
    double dE  = Pout_E - mP;
    double dpx = Pout_px;
    double dpy = Pout_py;
    double dpz = Pout_pz;
    double t   = dE*dE - (dpx*dpx + dpy*dpy + dpz*dpz);

    return {Q2, xB, W, t};
}

// ─── Format one particle line (compact, no -999) ──────────────────────────
struct ParticleLine {
    int status;
    int pid;
    int det;
    double p_gen, theta_gen, phi_gen, vz_gen;
    double p_rec, theta_rec, phi_rec, vz_rec;
};

static void write_particle_line(std::ostream& os, const ParticleLine& p)
{
    os << " " << p.status
       << "  " << std::setw(5) << std::right << p.pid
       << "  " << p.det << "  ";
    os << std::fixed;
    os << std::setw(8) << std::setprecision(4) << p.p_gen
       << "  " << std::setw(8) << std::setprecision(3) << p.theta_gen
       << "  " << std::setw(8) << std::setprecision(3) << p.phi_gen
       << "  " << std::setw(7) << std::setprecision(3) << p.vz_gen;
    if (p.status > 0) {
        os << "  " << std::setw(8) << std::setprecision(4) << p.p_rec
           << "  " << std::setw(8) << std::setprecision(3) << p.theta_rec
           << "  " << std::setw(8) << std::setprecision(3) << p.phi_rec
           << "  " << std::setw(7) << std::setprecision(3) << p.vz_rec;
    }
    os << "\n";
}

// ─── Header lines emitted at the top of each output file ──────────────────
static std::vector<std::string> make_header(const Params& p, const std::string& src_dir)
{
    auto mass_label = [](const std::string& r) -> std::string {
        if (r == "epK+K-")    return "M(K+K-)";
        if (r == "eppi+pi-")  return "M(pi+pi-)";
        if (r == "epe+e-")    return "M(e+e-)";
        return "M(hh)";
    };
    std::ostringstream oss;
    oss.str(""); oss << "#! reaction: " << p.reaction;       std::string l1 = oss.str();
    oss.str(""); oss << "#! beam: " << pid_short(p.beam_pid) << " (" << p.beam_pid << ")"; std::string l2 = oss.str();
    oss.str(""); oss << "#! beam_energy: " << p.beam_energy; std::string l3 = oss.str();
    oss.str(""); oss << "#! target: " << pid_short(p.target_pid) << " (" << p.target_pid << ")"; std::string l4 = oss.str();
    oss.str(""); oss << "#! source: " << src_dir;             std::string l5 = oss.str();
    oss.str(""); oss << "#! columns_event: event_num nrec Q2 xB W t " << mass_label(p.reaction); std::string l6 = oss.str();
    oss.str(""); oss << "#! matching: hadrons hit-based TruthMatch via MC::RecMatch quality>"
                     << p.quality_thresh << "; electron "
                     << (p.electron_match == "pidonly"
                            ? "PID==11 in FD nearest in p"
                            : "hit-based TruthMatch via MC::RecMatch quality>" + std::to_string(p.quality_thresh))
                     << "; species from MC::Particle.pid; gen electron theta>="
                     << p.min_electron_theta << " deg"; std::string l7 = oss.str();
    std::string l8 = "#! columns_particle: status(0=not_detected,1=matched_no_PID,2=matched_with_PID) "
                     "pid det(0=none,1=FD,2=CD,3=FT) p_gen theta_gen phi_gen vz_gen "
                     "[p_rec theta_rec phi_rec vz_rec]  # rec columns OMITTED when status==0";
    return {l1,l2,l3,l4,l5,l6,l7,l8};
}

// ─── List *.hipo files in a directory (sorted) ────────────────────────────
static std::vector<std::string> list_hipo(const std::string& dir)
{
    std::vector<std::string> out;
    for (const auto& ent : fs::directory_iterator(dir)) {
        if (ent.is_regular_file() && ent.path().extension() == ".hipo")
            out.push_back(ent.path().string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

// ─── CLI parsing ──────────────────────────────────────────────────────────
static bool parse_args(int argc, char** argv, Params& p)
{
    int i = 1;
    auto need = [&](const std::string& flag) -> std::string {
        if (i + 1 >= argc) {
            std::cerr << "Missing argument after " << flag << std::endl;
            std::exit(2);
        }
        return argv[++i];
    };
    while (i < argc) {
        std::string a = argv[i];
        if      (a == "--reaction")            p.reaction        = need(a);
        else if (a == "--basename")            p.basename        = need(a);
        else if (a == "--gated_suffix")        p.gated_suffix    = need(a);
        else if (a == "--electron_match")      p.electron_match  = need(a);
        else if (a == "--beam_energy")         p.beam_energy     = std::stod(need(a));
        else if (a == "--beam_pid")            p.beam_pid        = std::stoi(need(a));
        else if (a == "--target_pid")          p.target_pid      = std::stoi(need(a));
        else if (a == "--target_mass")         p.target_mass     = std::stod(need(a));
        else if (a == "--min_electron_theta")  p.min_electron_theta = std::stod(need(a));
        else if (a == "--quality")             p.quality_thresh  = std::stod(need(a));
        else if (a == "--val_fraction")        p.val_fraction    = std::stod(need(a));
        else if (a == "--max_files")           p.max_files       = std::stoi(need(a));
        else if (a == "--file_offset")         p.file_offset     = std::stoi(need(a));
        else if (a == "--seed")                p.seed            = std::stoul(need(a));
        else if (a == "-v" || a == "--verbose") p.verbose = true;
        else if (p.hipo_dir.empty())   p.hipo_dir   = a;
        else if (p.output_dir.empty()) p.output_dir = a;
        else { std::cerr << "Unknown arg: " << a << std::endl; return false; }
        ++i;
    }
    if (p.hipo_dir.empty() || p.output_dir.empty()) {
        std::cerr <<
        "Usage: " << argv[0] << " <hipo_dir> <output_dir> [options]\n"
        "  --reaction <r>             default: epK+K-\n"
        "  --basename <s>             default: phi_tm\n"
        "  --gated_suffix <s>         default: hadgated\n"
        "  --electron_match <mode>    pidonly|truth, default: pidonly\n"
        "  --beam_energy <GeV>        default: 10.6\n"
        "  --beam_pid <pid>           default: 11\n"
        "  --target_pid <pid>         default: 2212\n"
        "  --target_mass <GeV>        default: 0.938272\n"
        "  --min_electron_theta <deg> default: 6.0\n"
        "  --quality <q>              default: 0.98\n"
        "  --val_fraction <f>         default: 0.20\n"
        "  --max_files <N>            default: 0 (all)\n"
        "  --file_offset <N>          default: 0\n"
        "  --seed <N>                 default: 42\n"
        << std::endl;
        return false;
    }
    return true;
}

// ─── Main ────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    Params p;
    if (!parse_args(argc, argv, p)) return 2;

    // Save full command line to params.txt next to output_dir (in ../report/ if it exists, else output_dir)
    {
        fs::create_directories(p.output_dir);
        std::string params_dir = p.output_dir;
        fs::path report_dir = fs::path(p.output_dir).parent_path() / "report";
        if (fs::is_directory(report_dir)) params_dir = report_dir.string();
        std::ofstream pf(params_dir + "/params.txt");
        // params.txt — human-readable record
        pf << "# Command line:\n";
        for (int i = 0; i < argc; ++i) pf << argv[i] << (i < argc-1 ? " " : "\n");
        pf << "\n# Parameters:\n"
           << "hipo_dir:            " << p.hipo_dir << "\n"
           << "output_dir:          " << p.output_dir << "\n"
           << "reaction:            " << p.reaction << "\n"
           << "beam_pid:            " << p.beam_pid << "\n"
           << "beam_energy:         " << p.beam_energy << "\n"
           << "target_pid:          " << p.target_pid << "\n"
           << "quality_thresh:      " << p.quality_thresh << "\n"
           << "min_electron_theta:  " << p.min_electron_theta << "\n"
           << "electron_match:      " << p.electron_match << "\n"
           << "val_fraction:        " << p.val_fraction << "\n"
           << "max_files:           " << p.max_files << "\n"
           << "file_offset:         " << p.file_offset << "\n"
           << "seed:                " << p.seed << "\n"
           << "basename:            " << p.basename << "\n";

        // launch.sh — re-runnable script
        std::ofstream lf(params_dir + "/launch.sh");
        lf << "#!/bin/bash\n"
           << "# Auto-generated — reproduces this run exactly\n";
        for (int i = 0; i < argc; ++i) {
            std::string a = argv[i];
            // Use absolute path for the executable
            if (i == 0) a = fs::canonical(a).string();
            // Quote args with spaces or special chars
            if (a.find(' ') != std::string::npos || a.find('*') != std::string::npos)
                lf << "\"" << a << "\"";
            else
                lf << a;
            lf << (i < argc-1 ? " \\\n    " : "\n");
        }
        lf.close();
        // Make launch.sh executable
        chmod((params_dir + "/launch.sh").c_str(), 0755);
    }

    if (!fs::is_directory(p.hipo_dir)) {
        std::cerr << "Error: " << p.hipo_dir << " is not a directory" << std::endl;
        return 2;
    }
    fs::create_directories(p.output_dir);

    auto hipo_files = list_hipo(p.hipo_dir);
    if (hipo_files.empty()) {
        std::cerr << "Error: no *.hipo files in " << p.hipo_dir << std::endl;
        return 2;
    }
    const int n_total = hipo_files.size();
    if (p.file_offset > 0) hipo_files.erase(hipo_files.begin(), hipo_files.begin()+p.file_offset);
    if (p.max_files > 0 && (int)hipo_files.size() > p.max_files)
        hipo_files.resize(p.max_files);

    std::cout << "Found " << n_total << " HIPO files; processing "
              << hipo_files.size()
              << " (offset=" << p.file_offset << ", max=" << p.max_files << ")\n";

    const auto expected  = expected_pids(p.reaction);
    const auto canon_order = canonical_pid_order(p.reaction);
    const int  npart    = (int)expected.size();
    const auto header   = make_header(p, fs::absolute(p.hipo_dir).string());

    auto out_path = [&](const std::string& tag){
        return p.output_dir + "/" + p.basename + tag + ".dat";
    };

    std::ofstream f_train (out_path("_train"));
    std::ofstream f_val   (out_path("_val"));
    std::ofstream f_gtrain(out_path("_" + p.gated_suffix + "_train"));
    std::ofstream f_gval  (out_path("_" + p.gated_suffix + "_val"));

    for (auto* f : {&f_train, &f_val, &f_gtrain, &f_gval})
        for (const auto& h : header) *f << h << "\n";

    std::mt19937_64 rng(p.seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    long long event_num         = 0;
    long long n_total_kept      = 0;
    long long n_skipped_channel = 0;
    long long n_skipped_theta   = 0;
    long long n_train = 0, n_val = 0, n_gtrain = 0, n_gval = 0;
    long long n_e_fd_gated = 0;
    std::vector<long long> n_rec_per_particle(npart, 0);
    std::vector<long long> nrec_counts(npart + 1, 0);

    // ── Loop files ────────────────────────────────────────────────────────
    // Install our SIGABRT handler so a bad-HIPO-file vector assert doesn't
    // kill the whole run.
    std::signal(SIGABRT, abort_signal_handler);
    long long n_aborted = 0;

    int ifile = 0;
    const int nfiles = hipo_files.size();
    const int print_every = std::max(10, nfiles / 100);  // ~100 progress lines max, min every 10th
    for (const auto& hfile : hipo_files) {
        ++ifile;
        if (ifile == 1 || ifile % print_every == 0 || ifile == nfiles)
            std::cout << "  File " << ifile << "/" << nfiles
                      << " [events so far: " << event_num << "]\n" << std::flush;

        // If we land here with non-zero, we recovered from an abort in the
        // file body below.  Skip this file and move on.
        if (sigsetjmp(g_abort_jmp, 1) != 0) {
            std::cerr << "  [recovered from HIPO abort on this file]\n";
            ++n_aborted;
            // Re-install handler (some systems reset to SIG_DFL on delivery)
            std::signal(SIGABRT, abort_signal_handler);
            continue;
        }
        g_in_file = 1;

        try {
            hipo::reader  reader;
            reader.open(hfile.c_str());

            hipo::dictionary fact;
            reader.readDictionary(fact);

            hipo::bank bMc    (fact.getSchema("MC::Particle"));
            hipo::bank bRec   (fact.getSchema("REC::Particle"));
            hipo::bank bMatch (fact.getSchema("MC::RecMatch"));
            hipo::event ev;

            while (reader.next()) {
                reader.read(ev);
                ev.getStructure(bMc);
                ev.getStructure(bRec);
                ev.getStructure(bMatch);

                const int nmc = bMc.getRows();
                if (nmc < npart) continue;

                // Check the first `npart` MC PIDs match the expected set
                std::vector<int> got;  got.reserve(npart);
                for (int k = 0; k < npart; ++k) got.push_back(bMc.getInt("pid", k));
                std::vector<int> got_sorted = got;
                std::sort(got_sorted.begin(), got_sorted.end());
                if (got_sorted != expected) { ++n_skipped_channel; continue; }

                // Compute MC kinematics + identify electron index
                struct McInfo { int idx; int pid; double p, th, ph, vz, mass; };
                std::vector<McInfo> mc(npart);
                int idx_e = -1;
                for (int k = 0; k < npart; ++k) {
                    const Kin K = kinematics(bMc.getFloat("px", k),
                                             bMc.getFloat("py", k),
                                             bMc.getFloat("pz", k));
                    mc[k] = {k, got[k], K.p, K.theta, K.phi,
                             bMc.getFloat("vz", k), 0.0};
                    if (got[k] == 11) idx_e = k;
                }

                // Reorder mc to canonical order [e-, p, h+, h-] to match Python
                {
                    std::vector<McInfo> mc_reordered(npart);
                    for (int ci = 0; ci < npart; ++ci) {
                        int target_pid = canon_order[ci];
                        bool found = false;
                        for (int k = 0; k < npart; ++k) {
                            if (mc[k].pid == target_pid) {
                                mc_reordered[ci] = mc[k];
                                mc_reordered[ci].idx = mc[k].idx;  // keep original HIPO index for truth map
                                found = true;
                                // mark as used to handle duplicates (e.g. e+e- in epe+e-)
                                mc[k].pid = 0;
                                break;
                            }
                        }
                        if (!found) { /* shouldn't happen after channel check */ }
                    }
                    mc = mc_reordered;
                    idx_e = 0;  // electron is always index 0 in canonical order
                }

                // gen e- theta cut → drop event
                if (idx_e < 0 || mc[idx_e].th < p.min_electron_theta) {
                    ++n_skipped_theta;
                    continue;
                }

                // Build truth map: mc_idx -> (rec_idx, quality)
                std::unordered_map<int, std::pair<int,double>> truth;
                const int nmt = bMatch.getRows();
                for (int r = 0; r < nmt; ++r) {
                    int    rec_idx = bMatch.getInt  ("pindex", r);
                    int    mc_idx  = bMatch.getInt  ("mcindex", r);
                    double q       = bMatch.getFloat("quality", r);
                    auto it = truth.find(mc_idx);
                    if (it == truth.end() || q > it->second.second)
                        truth[mc_idx] = {rec_idx, q};
                }

                const int nrec = bRec.getRows();

                // For each MC particle, find its REC partner (if any)
                std::vector<ParticleLine> plines(npart);
                bool e_gated = false;
                int  nrec_found = 0;

                // ── Pre-pass for pidonly electron matching ───────────────
                // Scan REC::Particle for pid==11 in FD and pick the one
                // closest in p to the MC electron.  This is the v11 default.
                int  rec_idx_for_e = -1;     // -1 means "no match"
                if (p.electron_match == "pidonly" && idx_e >= 0) {
                    double best_dp = 1e30;
                    for (int r = 0; r < nrec; ++r) {
                        if (bRec.getInt("pid", r) != 11) continue;
                        int rec_st = bRec.getInt("status", r);
                        if (det_from_status(rec_st) != 1) continue;  // FD only
                        double rpx = bRec.getFloat("px", r);
                        double rpy = bRec.getFloat("py", r);
                        double rpz = bRec.getFloat("pz", r);
                        double rp  = std::sqrt(rpx*rpx + rpy*rpy + rpz*rpz);
                        double dp  = std::abs(rp - mc[idx_e].p);
                        if (dp < best_dp) { best_dp = dp; rec_idx_for_e = r; }
                    }
                }

                for (int k = 0; k < npart; ++k) {
                    ParticleLine& pl = plines[k];
                    pl.pid       = mc[k].pid;
                    pl.p_gen     = mc[k].p;
                    pl.theta_gen = mc[k].th;
                    pl.phi_gen   = mc[k].ph;
                    pl.vz_gen    = mc[k].vz;
                    pl.status    = 0;
                    pl.det       = 0;

                    // ── Determine which REC particle (if any) this MC pairs with
                    int rec_idx = -1;
                    if (mc[k].pid == 11 && p.electron_match == "pidonly") {
                        rec_idx = rec_idx_for_e;        // from pre-pass
                    } else {
                        // Default: TruthMatch via MC::RecMatch
                        // Use original HIPO index (mc[k].idx), not canonical index k
                        auto it = truth.find(mc[k].idx);
                        if (it != truth.end() && it->second.second > p.quality_thresh)
                            rec_idx = it->second.first;
                    }
                    if (rec_idx < 0 || rec_idx >= nrec) continue;

                    int    rec_pid = bRec.getInt  ("pid",    rec_idx);
                    int    rec_st  = bRec.getInt  ("status", rec_idx);
                    double rpx     = bRec.getFloat("px",     rec_idx);
                    double rpy     = bRec.getFloat("py",     rec_idx);
                    double rpz     = bRec.getFloat("pz",     rec_idx);
                    double rvz     = bRec.getFloat("vz",     rec_idx);
                    const Kin K    = kinematics(rpx, rpy, rpz);

                    pl.status    = (rec_pid == mc[k].pid) ? 2 : 1;
                    pl.det       = det_from_status(rec_st);
                    pl.p_rec     = K.p;
                    pl.theta_rec = K.theta;
                    pl.phi_rec   = K.phi;
                    pl.vz_rec    = rvz;

                    ++n_rec_per_particle[k];
                    ++nrec_found;
                    if (mc[k].pid == 11 && pl.det == 1) e_gated = true;
                }

                ++nrec_counts[std::min(nrec_found, npart)];

                // Event-line kinematics — compute from GEN (truth).  These are
                // metadata only; downstream training reads particle lines, not
                // event-line columns.
                EvtKin EK{0,0,0,0};
                {
                    Kin e_kin { mc[idx_e].p, mc[idx_e].th, mc[idx_e].ph };
                    Kin p_kin {0,0,0};
                    for (int k = 0; k < npart; ++k)
                        if (mc[k].pid == 2212)
                            p_kin = { mc[k].p, mc[k].th, mc[k].ph };
                    EK = compute_evt_kin(p.beam_energy, p.target_mass,
                                         e_kin, e_kin, p_kin, p_kin);
                }
                // M(hh) = invariant mass of the two non-electron, non-proton
                // hadrons (MC indices 2 and 3 in the standard ordering, e.g.
                // K+ and K- for the phi channel).
                double Mh = 0.0;
                if (npart >= 4) {
                    auto pid_mass = [](int pid)->double {
                        switch (std::abs(pid)) {
                            case 321:  return 0.49368;   // K
                            case 211:  return 0.13957;   // pi
                            case 11:   return 0.000511;  // e
                            case 13:   return 0.10566;   // mu
                            case 2212: return 0.938272;  // p
                            default:   return 0.0;
                        }
                    };
                    auto en = [](double pp, double mm){ return std::sqrt(pp*pp + mm*mm); };
                    double m1 = pid_mass(mc[2].pid);
                    double m2 = pid_mass(mc[3].pid);
                    double E1 = en(mc[2].p, m1);
                    double E2 = en(mc[3].p, m2);
                    double p1x = mc[2].p * std::sin(mc[2].th/DEG_PER_RAD) * std::cos(mc[2].ph/DEG_PER_RAD);
                    double p1y = mc[2].p * std::sin(mc[2].th/DEG_PER_RAD) * std::sin(mc[2].ph/DEG_PER_RAD);
                    double p1z = mc[2].p * std::cos(mc[2].th/DEG_PER_RAD);
                    double p2x = mc[3].p * std::sin(mc[3].th/DEG_PER_RAD) * std::cos(mc[3].ph/DEG_PER_RAD);
                    double p2y = mc[3].p * std::sin(mc[3].th/DEG_PER_RAD) * std::sin(mc[3].ph/DEG_PER_RAD);
                    double p2z = mc[3].p * std::cos(mc[3].th/DEG_PER_RAD);
                    double Mh2 = (E1+E2)*(E1+E2)
                               - ((p1x+p2x)*(p1x+p2x) + (p1y+p2y)*(p1y+p2y) + (p1z+p2z)*(p1z+p2z));
                    Mh = (Mh2 > 0) ? std::sqrt(Mh2) : 0.0;
                }

                std::ostringstream evtline;
                evtline << ++event_num << "  " << nrec_found
                        << std::fixed << std::setprecision(4)
                        << "  " << EK.Q2
                        << "  " << EK.xB
                        << "  " << EK.W
                        << "  " << EK.t
                        << "  " << Mh
                        << "\n";

                // Choose train/val split
                const bool is_val = (u01(rng) < p.val_fraction);
                std::ostream& f  = is_val ? f_val   : f_train;
                f << evtline.str();
                for (const auto& pl : plines) write_particle_line(f, pl);
                is_val ? ++n_val : ++n_train;

                if (e_gated) {
                    ++n_e_fd_gated;
                    std::ostream& fg = is_val ? f_gval : f_gtrain;
                    fg << evtline.str();
                    for (const auto& pl : plines) write_particle_line(fg, pl);
                    is_val ? ++n_gval : ++n_gtrain;
                }

                ++n_total_kept;
            }
        } catch (const std::exception& e) {
            std::cerr << "  [skip file due to exception: " << e.what() << "]\n";
        }
        g_in_file = 0;       // file done; any later abort should not jump here
    }

    // ── Summary ───────────────────────────────────────────────────────────
    std::cout << "\nSummary:\n"
              << "  Kept events:                       " << n_total_kept << "\n"
              << "  Skipped wrong channel:             " << n_skipped_channel << "\n"
              << "  Skipped gen e- theta < " << p.min_electron_theta << ": " << n_skipped_theta << "\n"
              << "  Files aborted (HIPO crashes):      " << n_aborted << "\n";
    for (int k = 0; k < npart; ++k) {
        std::cout << "    index " << k << " [" << std::setw(4) << pid_short(canon_order[k])
                  << "] matched: " << n_rec_per_particle[k] << "\n";
    }
    std::cout << "  Electron truth-matched in FD:      " << n_e_fd_gated << "\n";
    std::cout << "\n  Electron file:  " << out_path("_train") << " / " << out_path("_val")
              << "  (" << n_train << " / " << n_val << " events)\n";
    std::cout << "  Hadron-gated file:  " << out_path("_" + p.gated_suffix + "_train")
              << " / " << out_path("_" + p.gated_suffix + "_val")
              << "  (" << n_gtrain << " / " << n_gval << " events)\n";
    std::cout << "Done.\n";

    return 0;
}
