#pragma once
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#include <star/common/ArrayView.h>

namespace star::visualize
{

    /* GraphVisualizer, but templated
     */
    template <unsigned node_knn_size>
    inline void SaveGraph_Generic_Device(
        const GArrayView<float4> &vertices,
        const GArrayView<uchar3> &vertex_color,
        const GArrayView<unsigned> &vertex_weight,
        const GArrayView<float4> &normals,
        const GArrayView<float> &normal_weight,
        const GArrayView<ushortX<node_knn_size>> &edges,
        const GArrayView<floatX<node_knn_size>> &edge_weight,
        const std::string &path)
    {

        json graph_json;

        std::vector<float4> h_vertices;
        vertices.Download(h_vertices);

        std::vector<ushortX<node_knn_size>> h_edges;
        edges.Download(h_edges);

        // Vertex & edge
        std::vector<std::vector<float>> vec_vertices;
        std::vector<std::vector<int>> vec_edges;
        for (auto i = 0; i < h_vertices.size(); ++i)
        {
            // vertex
            float4 vertex = h_vertices[i];
            std::vector<float> v_coord{vertex.x, vertex.y, vertex.z};
            vec_vertices.push_back(v_coord);

            for (auto j = 0; j < node_knn_size; ++j)
            {
                int id_e = h_edges[i][j];
                if ((id_e >= 0) && (id_e < h_vertices.size()) && (id_e != i))
                {
                    std::vector<int> e_ind{i, id_e};
                    vec_edges.push_back(e_ind);
                }
            }
        }
        graph_json["vertices"] = vec_vertices;
        graph_json["edges"] = vec_edges;

        // Vertex weight
        if (vertex_weight.Size() != 0)
        {
            std::vector<unsigned> h_vertex_weight; // Variant here
            vertex_weight.Download(h_vertex_weight);
            std::vector<float> vec_vertices_weight;
            for (auto i = 0; i < h_vertices.size(); ++i)
            {
                // vertex
                float4 vertex = h_vertices[i];
                vec_vertices_weight.push_back((float)h_vertex_weight[i]);
            }
            graph_json["weight_v"] = vec_vertices_weight;
        }

        // Vertex Color;
        if (vertex_color.Size() != 0)
        {
            std::vector<uchar3> h_vertex_color;
            vertex_color.Download(h_vertex_color);
            std::vector<std::vector<float>> vec_vertices_color;
            for (auto i = 0; i < h_vertices.size(); ++i)
            {
                // vertex
                float4 vertex = h_vertices[i];
                uchar3 vertex_color = h_vertex_color[i];
                std::vector<float> v_color{
                    float(vertex_color.x) / 255.f,
                    float(vertex_color.y) / 255.f,
                    float(vertex_color.z) / 255.f};

                vec_vertices_color.push_back(v_color);
            }
            graph_json["color_v"] = vec_vertices_color;
        }

        // Normals
        if (normals.Size() != 0)
        {
            std::vector<float4> h_normals;
            normals.Download(h_normals);
            std::vector<std::vector<float>> vec_normals;
            for (auto i = 0; i < h_vertices.size(); ++i)
            {
                // vertex
                float4 vertex = h_vertices[i];
                float4 normal = h_normals[i];
                std::vector<float> n_direction{normal.x, normal.y, normal.z};
                vec_normals.push_back(n_direction);
            }
            graph_json["normals"] = vec_normals;
        }

        // Normal weight
        if (normal_weight.Size() != 0)
        {
            std::vector<float> h_normal_weight;
            normal_weight.Download(h_normal_weight);
            std::vector<float> vec_normal_weight;
            for (auto i = 0; i < h_vertices.size(); ++i)
            {
                // vertex
                float4 vertex = h_vertices[i];
                vec_normal_weight.push_back(h_normal_weight[i]);
            }
            graph_json["weight_n"] = vec_normal_weight;
        }

        // Edege weight
        if (edge_weight.Size() != 0)
        {
            std::vector<floatX<node_knn_size>> h_edge_weight;
            edge_weight.Download(h_edge_weight);
            std::vector<float> vec_edge_weight;
            for (auto i = 0; i < h_vertices.size(); ++i)
            {
                // vertex
                float4 vertex = h_vertices[i];
                for (auto j = 0; j < node_knn_size; ++j)
                {
                    int id_e = h_edges[i][j];
                    float w_e = h_edge_weight[i][j];
                    if ((id_e >= 0) && (id_e < h_vertices.size()) && (id_e != i))
                    {
                        vec_edge_weight.push_back(w_e);
                    }
                }
            }
            graph_json["weight_e"] = vec_edge_weight;
        }

        // Write out
        std::ofstream o(path);
        o << std::setw(4) << graph_json << std::endl;
        o.close();
    }

    template <unsigned node_knn_size>
    inline void SaveGraph_Generic_Host(
        const std::vector<float4> &h_vertices,
        const std::vector<uchar3> &h_vertex_color,
        const std::vector<unsigned> &h_vertex_weight,
        const std::vector<float4> &h_normals,
        const std::vector<float> &h_normal_weight,
        const std::vector<ushortX<node_knn_size>> &h_edges,
        const std::vector<floatX<node_knn_size>> &h_edge_weight,
        const std::string &path)
    {

        json graph_json;

        // Vertex & edge
        std::vector<std::vector<float>> vec_vertices;
        std::vector<std::vector<int>> vec_edges;
        for (auto i = 0; i < h_vertices.size(); ++i)
        {
            // vertex
            float4 vertex = h_vertices[i];
            std::vector<float> v_coord{vertex.x, vertex.y, vertex.z};
            vec_vertices.push_back(v_coord);

            for (auto j = 0; j < node_knn_size; ++j)
            {
                int id_e = h_edges[i][j];
                if ((id_e >= 0) && (id_e < h_vertices.size()) && (id_e != i))
                {
                    std::vector<int> e_ind{i, id_e};
                    vec_edges.push_back(e_ind);
                }
            }
        }
        graph_json["vertices"] = vec_vertices;
        graph_json["edges"] = vec_edges;

        // Vertex weight
        if (h_vertex_weight.size() != 0)
        {
            std::vector<float> vec_vertices_weight;
            for (auto i = 0; i < h_vertices.size(); ++i)
            {
                // vertex
                float4 vertex = h_vertices[i];
                vec_vertices_weight.push_back((float)h_vertex_weight[i]);
            }
            graph_json["weight_v"] = vec_vertices_weight;
        }

        // Vertex Color;
        if (h_vertex_color.size() != 0)
        {
            std::vector<std::vector<float>> vec_vertices_color;
            for (auto i = 0; i < h_vertices.size(); ++i)
            {
                // vertex
                float4 vertex = h_vertices[i];
                uchar3 vertex_color = h_vertex_color[i];
                std::vector<float> v_color{
                    float(vertex_color.x) / 255.f,
                    float(vertex_color.y) / 255.f,
                    float(vertex_color.z) / 255.f};

                vec_vertices_color.push_back(v_color);
            }
            graph_json["color_v"] = vec_vertices_color;
        }

        // Normals
        if (h_normals.size() != 0)
        {
            std::vector<std::vector<float>> vec_normals;
            for (auto i = 0; i < h_vertices.size(); ++i)
            {
                // vertex
                float4 vertex = h_vertices[i];
                float4 normal = h_normals[i];
                std::vector<float> n_direction{normal.x, normal.y, normal.z};
                vec_normals.push_back(n_direction);
            }
            graph_json["normals"] = vec_normals;
        }

        // Normal weight
        if (h_normal_weight.size() != 0)
        {
            std::vector<float> vec_normal_weight;
            for (auto i = 0; i < h_vertices.size(); ++i)
            {
                // vertex
                float4 vertex = h_vertices[i];
                vec_normal_weight.push_back(h_normal_weight[i]);
            }
            graph_json["weight_n"] = vec_normal_weight;
        }

        // Edege weight
        if (h_edge_weight.size() != 0)
        {
            std::vector<float> vec_edge_weight;
            for (auto i = 0; i < h_vertices.size(); ++i)
            {
                // vertex
                float4 vertex = h_vertices[i];
                for (auto j = 0; j < node_knn_size; ++j)
                {
                    int id_e = h_edges[i][j];
                    float w_e = h_edge_weight[i][j];
                    if ((id_e >= 0) && (id_e < h_vertices.size()) && (id_e != i))
                    {
                        vec_edge_weight.push_back(w_e);
                    }
                }
            }
            graph_json["weight_e"] = vec_edge_weight;
        }

        // Write out
        std::ofstream o(path);
        o << std::setw(4) << graph_json << std::endl;
        o.close();
    }

    /* Device Specifialize
     */
    template <unsigned node_knn_size>
    inline void SaveGraph(
        const GArrayView<float4> &vertices,
        const GArrayView<ushortX<node_knn_size>> &edges,
        const std::string &path)
    {
        SaveGraph_Generic_Device<node_knn_size>(
            vertices,
            GArrayView<uchar3>(),
            GArrayView<unsigned>(),
            GArrayView<float4>(),
            GArrayView<float>(),
            edges,
            GArrayView<floatX<node_knn_size>>(),
            path);
    }

    /* Host Specifialize
     */
    template <unsigned node_knn_size>
    inline void SaveGraph(
        const std::vector<float4> &h_vertices,
        const std::vector<uchar3> &h_vertex_color,
        const std::vector<ushortX<node_knn_size>> &h_edges,
        const std::string &path)
    {
        SaveGraph_Generic_Host<node_knn_size>(
            h_vertices,
            h_vertex_color,
            std::vector<unsigned>(),
            std::vector<float4>(),
            std::vector<float>(),
            h_edges,
            std::vector<floatX<node_knn_size>>(),
            path);
    }

    template <unsigned node_knn_size>
    inline void SaveGraph(
        const std::vector<float4> &h_vertices,
        const std::vector<uchar3> &h_vertex_color,
        const std::vector<ushortX<node_knn_size>> &h_edges,
        const std::vector<floatX<node_knn_size>> &h_edge_weight,
        const std::string &path)
    {
        SaveGraph_Generic_Host<node_knn_size>(
            h_vertices,
            h_vertex_color,
            std::vector<unsigned>(),
            std::vector<float4>(),
            std::vector<float>(),
            h_edges,
            h_edge_weight,
            path);
    }
}
