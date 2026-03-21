/**
 * Copyright (C) 2015 by Liangliang Nan (liangliang.nan@gmail.com)
 * https://3d.bk.tudelft.nl/liangliang/
 *
 * This file is part of Easy3D. If it is useful in your research/work,
 * I would be grateful if you show your appreciation by citing it:
 * ------------------------------------------------------------------
 *      Liangliang Nan.
 *      Easy3D: a lightweight, easy-to-use, and efficient C++
 *      library for processing and rendering 3D data. 2018.
 * ------------------------------------------------------------------
 * Easy3D is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License Version 3
 * as published by the Free Software Foundation.
 *
 * Easy3D is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "triangulation.h"
#include "matrix_algo.h"
#include <easy3d/optimizer/optimizer_lm.h>

using namespace easy3d;

//Global variables for use in non-linear optimization
Matrix34 M0;
Matrix34 M1;

Vector2D Point_data_0;
Vector2D Point_data_1;

class MyObjective : public Objective_LM {
public:
    MyObjective(int num_func, int num_var) : Objective_LM(num_func, num_var) {}
    /**
     *  Calculate the values of each function at x and return the function values as a vector in fvec.
     *  @param  x           The current values of variables.
     *  @param  fvec        Return the value vector of all the functions.
     *  @return Return a negative value to terminate.
     */
    int evaluate(const double *x, double *fvec) {
        Vector4D current_point(x[0],x[1],x[2],1);
        Vector3D projection_0 = M0*current_point;
        Vector3D projection_1 = M1*current_point;
        fvec[0] = projection_0.cartesian().x() - Point_data_0.x();
        fvec[1] = projection_0.cartesian().y() - Point_data_0.y();
        fvec[2] = projection_1.cartesian().x() - Point_data_1.x();
        fvec[3] = projection_1.cartesian().y() - Point_data_1.y();
        return 0;
    }
};

Matrix33 Normalization_Matrix(const std::vector<Vector2D>& points)
{
    //Calculate Mean
    double mean_x = 0;
    double mean_y = 0;
    int num_points = points.size();
    for (int i = 0; i < num_points; i++){
        mean_x += points[i].x();
        mean_y += points[i].y();
    }
    mean_x /= num_points;
    mean_y /= num_points;
    Vector2D mean_point(mean_x,mean_y);

    //Calculate mean distance from Mean
    double mean_dist = 0;
    for (int i = 0; i < num_points; i++){
        mean_dist += distance(points[i],mean_point);
    }
    mean_dist /= num_points;

    //Construct Transformation matrix
    double scale_factor = sqrt(2)/mean_dist;
    Matrix33 T( scale_factor, 0 ,           -mean_x*scale_factor,
                0,            scale_factor, -mean_y*scale_factor,
                0,            0 ,           1);

    return T;
}

std::vector<Vector2D> Normalize_points(
    const std::vector<Vector2D>& points,
    const Matrix33& norm_matrix){
    //Create copy with same shape
    std::vector<Vector2D> norm_points;
    //For each point norm_matrix * point = normalized point
    for (int i = 0; i < points.size(); i++){
        Vector3D point = points[i].homogeneous();
        Vector3D p = norm_matrix * point;
        norm_points.push_back(p);
    }
    return norm_points;
};

Matrix33 EstimateFundamentalMatrix(
    const std::vector<Vector2D>& points_0, /// input: 2D image points in the 1st image
    const std::vector<Vector2D>& points_1  /// input: 2D image points in the 2nd image
)
{
    // Set up matrix W
    int num_rows = points_0.size();
    int num_cols = 9;

    Matrix W(num_rows, num_cols, 0.0);
    for (int i = 0; i < num_rows; ++i)
    {
        double u0 = points_0[i].x();
        double v0 = points_0[i].y();
        double u1 = points_1[i].x();
        double v1 = points_1[i].y();
        W.set_row(i, {u0*u1, v0*u1, u1, u0*v1, v0*v1, v1, u0, v0, 1});
    }

    // Singular value decomposition of matrix W
    Matrix U(num_rows, num_rows, 0.0);
    Matrix S(num_rows, num_cols, 0.0);
    Matrix V(num_cols, num_cols, 0.0);

    svd_decompose(W, U, S, V);

    // Set up the Rank 3 version of F
    Vector f = V.get_column(V.cols() - 1);

    Matrix33 F_rank3(f[0],f[1],f[2],
                     f[3],f[4],f[5],
                     f[6],f[7],f[8]);
    Matrix33 U2;
    Matrix33 D2;
    Matrix33 V2;

    svd_decompose(F_rank3, U2, D2, V2);
    //Note that the singular values in D2 are already sorted in descending order per the description
    //in matrix_algo.h
    D2.set(2,2,0);

    //Calculate the rank 2 Matrix F
    Matrix33 F = U2 * D2 * V2.transpose();
    return F;
};

//Triangulation of Points
Vector3D TriangulatePoint(
    const Vector2D& p0,
    const Vector2D& p1,
    const Matrix34& M0,
    const Matrix34& M1
)
{
    Matrix A(4, 4, 0.0);

    for (int j = 0; j < 4; ++j) {
        A(0, j) = p0.x() * M0(2, j) - M0(0, j);
        A(1, j) = p0.y() * M0(2, j) - M0(1, j);
        A(2, j) = p1.x() * M1(2, j) - M1(0, j);
        A(3, j) = p1.y() * M1(2, j) - M1(1, j);
    }
    Matrix U(4, 4, 0.0);
    Matrix S(4, 4, 0.0);
    Matrix V(4, 4, 0.0);
    svd_decompose(A, U, S, V);

    Vector Xh = V.get_column(3);

    if (std::abs(Xh[3]) < 1e-12)
        return Vector3D(0.0, 0.0, 0.0);

    return Vector3D(
        Xh[0] / Xh[3],
        Xh[1] / Xh[3],
        Xh[2] / Xh[3]
    );
}

std::vector<Vector3D> TriangulateAllPoints(
    const std::vector<Vector2D>& points_0,
    const std::vector<Vector2D>& points_1,
    const Matrix33& K,
    const Matrix33& R,
    const Vector3D& t
)

{
    std::vector<Vector3D> reconstructed_points;

    Matrix34 P0_cam(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0
    );
    Matrix34 P0 = K * P0_cam;

    Matrix34 P1_cam(
        R(0, 0), R(0, 1), R(0, 2), t[0],
        R(1, 0), R(1, 1), R(1, 2), t[1],
        R(2, 0), R(2, 1), R(2, 2), t[2]
    );
    Matrix34 P1 = K * P1_cam;

    for (int i = 0; i < points_0.size(); ++i) {
        Vector3D X = TriangulatePoint(points_0[i], points_1[i], P0, P1);
        reconstructed_points.push_back(X);
    }
    return reconstructed_points;
}
/**
 * @return True on success, otherwise false. On success, the reconstructed 3D points must be written to 'points_3d'
 *      and the recovered relative pose must be written to R and t.
 */
bool Triangulation::triangulation(
        double fx, double fy,     /// input: the focal lengths (same for both cameras)
        double cx, double cy,     /// input: the principal point (same for both cameras)
        double s,                 /// input: the skew factor (same for both cameras)
        const std::vector<Vector2D> &points_0,  /// input: 2D image points in the 1st image.
        const std::vector<Vector2D> &points_1,  /// input: 2D image points in the 2nd image.
        std::vector<Vector3D> &points_3d,       /// output: reconstructed 3D points
        Matrix33 &R,   /// output: 3 by 3 matrix, which is the recovered rotation of the 2nd camera
        Vector3D &t    /// output: 3D vector, which is the recovered translation of the 2nd camera
) const {

    // Input validity check
    if (points_0.size() != points_1.size()){
        std::cerr << "Sizes of Matching points don't match" << std::endl;
        return false;
    }

    if (points_0.size() < 8){
        std::cerr << "At least 8 corresponding points needed" << std::endl;
        return false;
    }

    //Normalize the points and obtain the Matrices needed for denormalization
    Matrix33 T0 = Normalization_Matrix(points_0);
    std::vector<Vector2D> norm_points_0 = Normalize_points(points_0, T0);
    Matrix33 T1 = Normalization_Matrix(points_1);
    std::vector<Vector2D> norm_points_1 = Normalize_points(points_1, T1);

    //Estimate the normalized fundamental matrix and denormalize it
    Matrix33 F_norm = EstimateFundamentalMatrix(norm_points_0, norm_points_1);
    Matrix33 F = T1.transpose() * F_norm * T0;

    //Construct the matrix K, same for both cameras
    Matrix33 K(fx, s, cx,
               0, fy, cy,
               0,  0,  1);
    //Calculate Essential Matrix E
    Matrix33 E = K.transpose() * F * K;

    Matrix33 W( 0,-1,0,
                1,0,0,
                0,0,1);

    //SVD of E for recovery R and T
    Matrix33 U;
    Matrix33 D;
    Matrix33 V;
    svd_decompose(E,U,D,V);
    Matrix33 UWV_T = U*W*V.transpose();
    Matrix33 UW_TV_T = U*W.transpose()*V.transpose();

    Matrix33 R1 = determinant(UWV_T)*UWV_T;
    Matrix33 R2 = determinant(UW_TV_T)*UW_TV_T;

    Vector3D t1 = U.get_column(2);
    Vector3D t2 = -U.get_column(2);

    //Orientation check
    std::vector<Vector3D> points_3d_r1t1;
    std::vector<Vector3D> points_3d_r1t2;
    std::vector<Vector3D> points_3d_r2t1;
    std::vector<Vector3D> points_3d_r2t2;

    points_3d_r1t1 = TriangulateAllPoints(points_0,points_1,K,R1,t1);
    points_3d_r1t2 = TriangulateAllPoints(points_0,points_1,K,R1,t2);
    points_3d_r2t1 = TriangulateAllPoints(points_0,points_1,K,R2,t1);
    points_3d_r2t2 = TriangulateAllPoints(points_0,points_1,K,R2,t2);

    int count[4]={0};

    for(int i = 0; i < points_3d_r1t1.size(); ++i) {
        Vector3D X = points_3d_r1t1[i];
        // Check 1: Front of camera 1 ?
        if (X.z() <= 0) {
            continue;
        }
        // Check 2: Front of Camera 2?
        Vector3D X_c2 = R1 * X + t1;
        if (X_c2.z() > 0) {
            count[0]++;
        }
    }

    for(int i = 0; i < points_3d_r1t2.size(); ++i) {
        Vector3D X = points_3d_r1t2[i];
        // Check 1: Front of camera 1 ?
        if (X.z() <= 0) {
            continue;
        }
        // Check 2: Front of Camera 2?
        Vector3D X_c2 = R1 * X + t2;
        if (X_c2.z() > 0) {
            count[1]++;
        }
    }

    for(int i = 0; i < points_3d_r2t1.size(); ++i) {
        Vector3D X = points_3d_r2t1[i];
        // Check 1: Front of camera 1 ?
        if (X.z() <= 0) {
            continue;
        }
        // Check 2: Front of Camera 2?
        Vector3D X_c2 = R2 * X + t1;
        if (X_c2.z() > 0) {
            count[2]++;
        }
    }

    for(int i = 0; i < points_3d_r2t2.size(); ++i) {
        Vector3D X = points_3d_r2t2[i];
        // Check 1: Front of camera 1 ?
        if (X.z() <= 0) {
            continue;
        }
        // Check 2: Front of Camera 2?
        Vector3D X_c2 = R2 * X + t2;
        if (X_c2.z() > 0) {
            count[3]++;
        }
    }
    //Counting the votes
    int idx=0,big=0;
    for(int i=0;i<4;i++) {
        if (count[i] > big) {
            big = count[i];
            idx = i;
        }
    }
    //Assignment of variables
    if (idx==0) {
        points_3d=points_3d_r1t1;
        R=R1;
        t=t1;
    }

    else if (idx==1) {
        points_3d=points_3d_r1t2;
        R=R1;
        t=t2;
    }
    else if (idx==2) {
        points_3d=points_3d_r2t1;
        R=R2;
        t=t1;
    }
    else if (idx==3) {
        points_3d=points_3d_r2t2;
        R=R2;
        t=t2;
    }

    std::cout<<"\nR1 "<<R1<<std::endl;
    std::cout<<"\nR2 "<<R2<<std::endl;
    std::cout<<"\nt1 "<<t1<<std::endl;
    std::cout<<"\nt2 "<<t2<<std::endl;
    std::cout<<"\nR1t1 "<<count[0];
    std::cout<<"\nR1t2 "<<count[1];
    std::cout<<"\nR2t1 "<<count[2];
    std::cout<<"\nR2t2 "<<count[3]<<std::endl;

    // Explicit calculation of M0 and M1
    Matrix34 P0_cam(
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0
    );
    M0 = K * P0_cam;

    Matrix34 P1_cam(
        R(0, 0), R(0, 1), R(0, 2), t[0],
        R(1, 0), R(1, 1), R(1, 2), t[1],
        R(2, 0), R(2, 1), R(2, 2), t[2]
    );
    M1 = K * P1_cam;

    //Non-linear least squares using Levenberg Marquardt
    for (int i=0;i<points_3d.size();i++)
    {
        MyObjective obj(4,3);
        Optimizer_LM lm;
        std::vector<double> x = {points_3d[i].x(),points_3d[i].y(),points_3d[i].z()};
        Point_data_0 = points_0[i];
        Point_data_1 = points_1[i];

        bool status = lm.optimize(&obj,x);
        Vector3D new_point(x[0],x[1],x[2]);
        points_3d[i] = new_point;
    }

    //Error Calculation
    double total_squared_error=0.0;
    int valid_points=0;

    for (int i=0;i<points_3d.size();i++) {
        Vector3D X = points_3d[i];
        Vector4D X_homogeneous = X.homogeneous();

        Vector3D p0_homogeneous = M0 * X_homogeneous;
        Vector3D p1_homogeneous = M1 * X_homogeneous;

        if(std::abs(p1_homogeneous.z()) > 1e-12 && p0_homogeneous.z() > 1e-12) {
            Vector2D p0_reprojected = p0_homogeneous.cartesian();
            Vector2D p1_reprojected = p1_homogeneous.cartesian();

            double error_0 = distance(p0_reprojected, points_0[i]);
            double error_1 = distance(p1_reprojected, points_1[i]);
            total_squared_error += error_0*error_0 + error_1*error_1;
            valid_points++;
        }
    }

    double mean_squared_error = total_squared_error / (2 * valid_points);
    double RMSE = std::sqrt(mean_squared_error);
    std::cout << "RMSE: " << RMSE << " pixels" << std::endl;

    return true;
}
