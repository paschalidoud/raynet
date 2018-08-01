"""Contains various general purpose functions used to parse information for the
ground-truth data as well as for the bounding box that encloses the scene
"""
import os

import numpy as np
from scipy.io import loadmat
import xml.etree.ElementTree as ET

from ..utils.training_utils import get_triangles


def parse_scene_info(scene_info_filename):
    """ Parses the scene_info xml file in order take the information regarding
    the bounding box of each scene.

    Return:
    ------
        bbox: 1x6 numpy array, dtype=np.float32
              The bounding box that encloses the scene. The first three
              elements hold the minumum values and the latter the maximum
    """
    tree = ET.parse(scene_info_filename)
    root = tree.getroot()

    # Save exported information in a dictionary (of dictionaries) containing
    # information regarding the bounding box of each scene, the resolution as
    # well as the number of swallow trees used to represent the volume.
    d = {}
    for child in root:
        d[child.tag] = child.attrib

    # For now return only the information regarding the bounding box as a
    # dictionary with values "maxx", "maxy", "maxz", "minx", "miny", "minz"
    bbox = d["bbox"]
    return np.array([
        [bbox["minx"], bbox["miny"], bbox["minz"]],
        [bbox["maxx"], bbox["maxy"], bbox["maxz"]]
    ], dtype=np.float32).reshape(1, -1)


def parse_scene_info_dtu_dataset(scene_file):
    """Given an input file containing the bounding box of the scene return the
    a 1x6 numpy array containing this bbox.

    Arguments:
    ----------
        scene_files: str, Path to the file
    Return:
    -------
        bbox: 1x6 array, dtype=np.float32
              The bounding box that encloses the scene. The first three
              elements hold the minumum values and the latter the maximum
    """
    # Load the mat file containing various information for the scene
    scene_info = loadmat(scene_file, squeeze_me=True)
    bbox = scene_info["BB"].astype(np.float32)
    return bbox.reshape(1, -1)


def parse_gt_data_from_ply(gt_file):
    """ Given a file in ply format containing the ground truth data parse and
    return all the necessary information regarding the vertices, the normals
    and the faces of the scene
    """
    lines = []
    with open(gt_file, "r") as f:
        # Firstly read (and ignore) 13 lines containing the header information
        for i in range(13):
            x = f.readline()
            # Find the line containing the total number of vertices in the
            # file. This line will have the following format "element vertex
            # num_vertexes"
            if "element vertex" in x:
                num_vertices = int(x.strip().split(" ")[-1])
        # Read the data from the file and save them into a list
        lines = f.readlines()
    f.close()

    # The gt_file contains both points and faces. The points are organized in
    # three numbers, whereas the faces contain the indexes of the corresponding
    # points
    data = [x.strip().split(" ") for x in lines][:num_vertices]
    # Convert strings to float
    data = [map(float, x) for x in data]
    D = np.array(data, dtype=np.float32)

    faces = [x.strip().split(" ") for x in lines][num_vertices:]
    # Convert strings to int because they are indices
    data = [map(int, x) for x in faces]
    faces_idxs = np.array(data)[:, 1:]

    # Return three arrays containing points, normals and faces (as points
    # indices) respectively
    return D[:, 0:3], D[:, 3:], faces_idxs


def parse_gt_data_from_obj(gt_file):
    """ Given a file in obj format containing the ground truth data parse and
    return all the necessary information regarding the vertices, the normals
    and the faces of the scene
    """
    lines = []
    with open(gt_file, "r") as f:
        lines = f.readlines()
    f.close()

    # Create lists for vertices, normals and faces
    v = []
    vn = []
    f = []

    # An OBJ file contains a list of geometric vertices, which start with the
    # letter v, a list of vertex normals which start with the letter vn and a
    # list of polygonal face elements which start with with the letter f
    for l in lines:
        if l.startswith("v "):
            v.append(l)
        elif l.startswith("vn "):
            vn.append(l)
        elif l.startswith("f"):
            f.append(l)
    # Get the vertices and the normals and turn them to floats
    t = [i.strip().split(" ")[1:] for i in v]
    vertices = np.array([map(float, x) for x in t], dtype=np.float32)
    t = [i.strip().split(" ")[1:] for i in vn]
    normals = np.array([map(float, x) for x in t], dtype=np.float32)

    t = [i.strip().split(" ")[1:] for i in f]
    faces_idxs = []
    for tt in t:
        faces_idxs.append([i.split("//")[0] for i in tt])
    faces_idxs = np.array([map(int, x) for x in faces_idxs])
    # Remove 1 to make it compatible with the zero notation
    faces_idxs = faces_idxs - 1

    return vertices, normals, faces_idxs


def parse_gt_data(input_directory):
    files = os.listdir(input_directory)
    if "gt_mesh.obj" in files:
        gt_file = os.path.join(input_directory, "gt_mesh.obj")
        # Given a ground truth mesh file parse all the necessary information
        # regarding points, normals and faces
        points, normals, faces = parse_gt_data_from_obj(gt_file)
    else:
        gt_file = os.path.join(input_directory, "gt_mesh.ply")
        # Given a ground truth mesh file parse all the necessary information
        # regarding points, normals and faces
        points, normals, faces = parse_gt_data_from_ply(gt_file)

    return points, normals, faces


def parse_gt_mesh(input_directory):
    points, normals, faces = parse_gt_data(input_directory)
    triangles = get_triangles(points, faces)

    return triangles


class PLYHeader(object):
    """Parse a PLY file header into an object"""
    class Element(object):
        def __init__(self, name, count, properties):
            assert len(properties) > 0
            self.name = name
            self.count = count
            self.properties = properties

        @property
        def bytes(self):
            return sum(p.bytes for p in self.properties)

    class Property(object):
        def __init__(self, name, type):
            self.name = name
            self.type = type

        @property
        def bytes(self):
            return {
                "float": 4,
                "uchar": 1,
                "int": 4
            }[self.type]

    def __init__(self, fileobj):
        assert fileobj.readline().strip() == "ply"

        lines = []
        while True:
            l = fileobj.readline()
            if "end_header" in l:
                break
            lines.append(l)

        # Version and format
        identifier, format, version = lines[0].split()
        assert identifier == "format"
        self.is_ascii = "ascii" in format
        self.version = float(version)
        self.little_endian = "little" in format
        lines.pop(0)

        # Comments
        self.comments = [
            x.split(" ", 1)[1]
            for x in lines
            if x.startswith("comment")
        ]

        # Elements
        lines = [l for l in lines if not l.startswith("comment")]
        elements = []
        while lines:
            identifier, name, count = lines[0].split()
            assert identifier == "element"
            count = int(count)
            lines.pop(0)

            properties = []
            while lines:
                identifier, type, name = lines[0].split()
                if identifier != "property":
                    break
                properties.append(self.Property(name, type))
                lines.pop(0)
            elements.append(self.Element(name, count, properties))
        self.elements = elements


def parse_stl_file_to_pointcloud(stl_file):
    with open(stl_file, "rb") as f:
        header = PLYHeader(f)
        assert len(header.elements) == 1
        el = header.elements[0]
        assert all(p.type == "float" for p in el.properties[:3])

        # Read the data and place one element per line and skip all the extra
        # elements
        data = np.fromfile(f, dtype=np.uint8)
        data = data.reshape(-1, header.elements[0].bytes)
        data = data[:, :sum(p.bytes for p in el.properties[:3])].ravel()

        # Reread in the correct byte-order
        order = "<" if header.little_endian else ">"
        dtype = order + "f4"
        points = np.frombuffer(data.data, dtype=dtype).reshape(-1, 3)

        return points
