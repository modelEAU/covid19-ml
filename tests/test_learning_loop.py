import sys

import pytest
from pydantic import BaseModel

sys.path.append("src")
from learning_loop import model_to_dict


def test_simple_base_model_conversion():
    class SampleModel(BaseModel):
        name: str

    obj = SampleModel(name="test")
    result = model_to_dict(obj)
    assert result == {"name": "test"}


def test_nested_base_model_conversion():
    class ChildModel(BaseModel):
        name: str

    class ParentModel(BaseModel):
        child: ChildModel

    child_obj = ChildModel(name="child_test")
    parent_obj = ParentModel(child=child_obj)

    result = model_to_dict(parent_obj)
    assert result == {"child": {"name": "child_test"}}


def test_list_of_base_models_conversion():
    class SampleModel(BaseModel):
        name: str

    obj_list = [SampleModel(name=f"test_{i}") for i in range(3)]
    result = model_to_dict(obj_list)
    assert result == [{"name": "test_0"}, {"name": "test_1"}, {"name": "test_2"}]


def test_primitive_types_conversion():
    assert model_to_dict(123) == 123
    assert model_to_dict("test_string") == "test_string"


def test_dict_conversion():
    sample_dict = {"key": "value", "num": 42}
    assert model_to_dict(sample_dict) == sample_dict
