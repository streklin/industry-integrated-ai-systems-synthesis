from mgraph_db.mgraph.MGraph import MGraph
from mgraph_db.mgraph.schemas.Schema__MGraph__Node import Schema__MGraph__Node
from mgraph_db.mgraph.schemas.Schema__MGraph__Node__Data import Schema__MGraph__Node__Data
from mgraph_db.mgraph.schemas.Schema__MGraph__Edge import Schema__MGraph__Edge, Schema__MGraph__Edge__Data
from mgraph_db.mgraph.schemas.Schema__MGraph__Graph import Schema__MGraph__Graph
from osbot_utils.type_safe.primitives.domains.identifiers.Edge_Id import Edge_Id
from osbot_utils.type_safe.primitives.domains.identifiers.Node_Id import Node_Id
from typing import Dict, List
from pydantic import BaseModel, Field


########################################################################
# Pydantic Objects
########################################################################

class NamedEntity(BaseModel):
    name: str = Field(description="Name of the entity extracted from the plot")
    description: str = Field(description="Description of the extracted entity")

class GraphTriplet(BaseModel):
    subject: NamedEntity = Field(description="The subject of the relationship")
    predicate: str = Field(description="The predicate of the relationship")
    object: NamedEntity = Field(description="The object of the relationship")



########################################################################
# MGraph Objects
########################################################################


# Custom Node for MGraph
class Custom_Node_Data(Schema__MGraph__Node__Data):
    name: str
    type: str
    description: str
    
class Custom_Node(Schema__MGraph__Node):
    node_data: Custom_Node_Data  # type: ignore
 

# Custom Edge for MGraph
class Custom_Edge_Data(Schema__MGraph__Edge__Data):
    predicate: str = ""  # This allows the 'predicate' key inside edge_data

class Custom_Edge(Schema__MGraph__Edge):
    edge_data: Custom_Edge_Data  # type: ignore


# Custom Graph Schema to preserve Custom_Node and Custom_Edge types when loading
class Custom_Graph(Schema__MGraph__Graph):
    nodes: Dict[Node_Id, Custom_Node]  # type: ignore
    edges: Dict[Edge_Id, Custom_Edge]  # type: ignore


class MGraphManager:
    """Manager class for handling MGraph operations."""

    def __init__(self):
        self.mgraph = MGraph()
    
    def restore_graph(self, graph_file="knowledge_graph.json"):
        """Restores the MGraph from a JSON file, ensuring that the custom node and edge types are preserved."""
        print(f"Loading knowledge graph from {graph_file}...")
        with open(graph_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        graph_schema = Custom_Graph.from_json(data)
        self.mgraph.graph.model.data = graph_schema  # type: ignore
        self.mgraph.edit().rebuild_index()

    def insert_triplet_list(self, relationships: list[GraphTriplet]):
        """Inserts a list of triplets into the MGraph."""
        
        # We build the KG from unique Entity names, so we need to keep track of which entities we've already added to the graph to avoid duplicates.
        # We can use a dictionary to map entity names to their corresponding node IDs in the graph, which will allow us to easily reference existing nodes when adding relationships.
        
        print(f"Inserting {len(relationships)} triplets into graph...")
        entities = {}

        with self.mgraph.edit() as edit:
            for triplet in relationships:
                
                subject_id = None
                object_id = None

                # insert base entites as nodes in the graph if they haven't already been added, and keep track of their node IDs in the entities dictionary
                if triplet.subject.name in entities:
                    subject_id = entities[triplet.subject.name]
                else:

                    subject = edit.new_node(
                        node_type=Custom_Node, # type: ignore
                        name=triplet.subject.name,
                        description=triplet.subject.description
                    )

                    subject_id = subject.node_id
                    entities[triplet.subject.name] = subject_id

                if triplet.object.name in entities:
                    object_id = entities[triplet.object.name]
                else:
                    object = edit.new_node(
                        node_type=Custom_Node, # type: ignore
                        name=triplet.object.name,
                        description=triplet.object.description
                    )
                    object_id = object.node_id
                    entities[triplet.object.name] = object_id
                
                # insert relationship as an edge in the graph, referencing the node IDs of the subject and object
                edit.new_edge(
                    edge_type=Custom_Edge,      # Tells mgraph to use your new schema
                    from_node_id=subject_id,
                    to_node_id=object_id,
                    edge_data={
                        "predicate": triplet.predicate  # This will now bypass the type-checker!
                    }
                )

    def export_graph(self, output_file="exported_graph.json"):
        """Exports the current state of the MGraph to a JSON file."""
        with self.mgraph.export() as export:
            data = export.to__mgraph_json()
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    def query_by_entity_type(self, entity_type: str):
        """
        Queries the knowledge graph for all entities of a specific type and their relationships.

        Args:
            entity_type: The type of entities to query for (e.g., character, location, object, event, theme, genre).
        Returns:
            A summary of all entities of the specified type and their relationships in the knowledge graph.
        """
        print(f"Querying graph for entities of type: {entity_type}")
        results = []

        with self.mgraph.data() as data:
            for node in data.nodes():
                if hasattr(node, 'node_data') and getattr(node.node_data, 'type', None) == entity_type:
                    results.append(getattr(node.node_data, 'name', None))

        return results

    def query_by_predicate(self, predicate: str):
        """
        Queries the knowledge graph for all edges of a specific predicate.

        Args:
            predicate: The predicate to query for (e.g., character, location, object, event, theme, genre).
        Returns:
            A summary of all edges of the specified predicate in the knowledge graph.
        """
        print(f"Querying graph for edges with predicate: {predicate}")
        results = []

        with self.mgraph.data() as data:
            for edge in data.edges():
                if getattr(edge.edge.data.edge_data, 'predicate', None) == predicate:
                    results.append((edge.from_node().node_data.name, edge.edge.data.edge_data.predicate, edge.to_node().node_data.name))

        return results

    def query_all(self):
        """
        Queries the knowledge graph for all entities and their relationships.

        Returns:
            A summary of all entities and their relationships in the knowledge graph.
        """
        results = []

        with self.mgraph.data() as data:
            for node in data.nodes():
                results.append(getattr(node.node_data, 'name', None))
            for edge in data.edges():
                results.append((edge.from_node().node_data.name, edge.edge.data.edge_data.predicate, edge.to_node().node_data.name))

        return results

    def query_by_entity_name(self, entity_name: str):
        """
        Queries the knowledge graph for a specific entity and its relationships.

        Args:
            entity_name: The name of the entity to query for.

        Returns:
            A summary of the specified entity and its relationships in the knowledge graph.
        """
        print(f"Querying graph for entities of name: {entity_name}")

        results = []

        target_node_id = None

        with self.mgraph.data() as data:
            for node in data.nodes():
                if hasattr(node, 'node_data') and getattr(node.node_data, 'name', None) == entity_name:
                    target_node_id = node.node_id
                    break    
        
            for edge in data.edges():
                if edge.from_node_id() == target_node_id or edge.to_node_id() == target_node_id:
            
                    subject = data.node(edge.from_node_id()).node_data.name
                    predicate = getattr(edge.edge.data.edge_data, 'predicate', None)
                    object = data.node(edge.to_node_id()).node_data.name
            
                    results.append((subject, predicate, object))

        return str(results)
    

    def insert_predicate(self, subject: str, predicate: str, object: str):
        """
        Inserts a new triplet into the knowledge graph, creating nodes for the subject and object if they do not already exist, and an edge for the predicate that connects them.

        Args:
            subject: The subject of the relationship to insert into the knowledge graph.
            predicate: The predicate of the relationship to insert into the knowledge graph.
            object: The object of the relationship to insert into the knowledge graph.

        Returns:
            True if task completed, False if an error ocurred.
        """

        print(f"Inserting triplet {subject}->{predicate}->{object}")

        try:
            subject_node_id = None
            object_node_id = None

            with self.mgraph.data() as data:
                for node in data.nodes():
                    if hasattr(node, 'node_data'):
                        if getattr(node.node_data, 'name', None) == subject:
                            subject_node_id = node.node_id
                        elif getattr(node.node_data, 'name', None) == object:
                            object_node_id = node.node_id
            
            with self.mgraph.edit() as edit:
                if subject_node_id is None:
                    subject_node = edit.new_node(
                        node_type=Custom_Node, # type: ignore
                        name=subject,
                        description=""
                    )
                    subject_node_id = subject_node.node_id

                if object_node_id is None:
                    object_node = edit.new_node(
                        node_type=Custom_Node, # type: ignore
                        name=object,
                        description=""
                    )
                    object_node_id = object_node.node_id

                edit.new_edge(
                    edge_type=Custom_Edge,      # Tells mgraph to use your new schema
                    from_node_id=subject_node_id,
                    to_node_id=object_node_id,
                    edge_data={
                        "predicate": predicate  # This will now bypass the type-checker!
                    }
                )
                return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def remove_predicate(self, subject: str, predicate: str, object: str):
        """
        Removes a specific predicate (relationship) between two entities in the knowledge graph.

        Args:
            subject: The subject of the relationship to remove from the knowledge graph.
            predicate: The predicate of the relationship to remove from the knowledge graph.
            object: The object of the relationship to remove from the knowledge graph.
        """
        print(f"Removing triplet {subject}->{predicate}->{object}")


        try:

            with self.mgraph.edit() as edit:
                for edge in edit.edges():  # type: ignore
                    edge_subject = edit.node(edge.from_node_id()).node_data.name  # type: ignore
                    edge_predicate = getattr(edge.edge_data, 'predicate', None)
                    edge_object = edit.node(edge.to_node_id()).node_data.name  # type: ignore

                    if edge_subject == subject and edge_predicate == predicate and edge_object == object:
                        edit.delete_edge(edge.edge_id)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False