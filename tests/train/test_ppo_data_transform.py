from types import SimpleNamespace

from datasets import Dataset as HfDataset

from swift.pipelines.train.dataset_transform import apply_ppo_data_transform


def _build_args(**kwargs):
    defaults = {
        'ppo_data_transform': 'none',
        'ppo_data_answer_key': 'answer',
        'ppo_data_judge_key': 'expect_acc',
        'ppo_data_judge_threshold': 0.5,
        'ppo_data_score_keys': None,
        'ppo_data_score_weights': None,
        'ppo_data_teacher_prompt': None,
        'ppo_data_label_key': None,
        'ppo_data_include_label_in_teacher_prompt': False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_ppo_data_transform_sft():
    dataset = HfDataset.from_list([
        {
            'messages': [{'role': 'user', 'content': '2+2=?'}],
            'answer': '4',
            'expect_acc': 0.8,
        },
        {
            'messages': [{'role': 'user', 'content': '3+3=?'}],
            'answer': '5',
            'expect_acc': 0.2,
        },
    ])

    transformed = apply_ppo_data_transform(dataset, _build_args(ppo_data_transform='sft'), split='train')

    assert len(transformed) == 1
    assert transformed[0]['messages'][-1] == {'role': 'assistant', 'content': '4'}


def test_ppo_data_transform_dpo():
    dataset = HfDataset.from_list([
        {
            'messages': [{'role': 'user', 'content': '1+1=?'}],
            'answer': '2',
            'expect_acc': 0.9,
            'llm_score': 0.2,
        },
        {
            'messages': [{'role': 'user', 'content': '1+1=?'}],
            'answer': '3',
            'expect_acc': 0.1,
            'llm_score': 0.0,
        },
        {
            'messages': [{'role': 'user', 'content': '2+2=?'}],
            'answer': '4',
            'expect_acc': 0.7,
            'llm_score': 0.1,
        },
    ])

    transformed = apply_ppo_data_transform(
        dataset,
        _build_args(
            ppo_data_transform='dpo',
            ppo_data_score_keys=['expect_acc', 'llm_score'],
            ppo_data_score_weights=[1.0, 0.1],
        ),
        split='train')

    assert len(transformed) == 1
    assert transformed[0]['messages'][-1]['content'] == '2'
    assert transformed[0]['rejected_response'] == '3'


def test_ppo_data_transform_opsd():
    dataset = HfDataset.from_list([
        {
            'messages': [{'role': 'user', 'content': 'What is 5+5?'}],
            'answer': '10',
            'expect_acc': 0.95,
        },
        {
            'messages': [{'role': 'user', 'content': 'What is 7+7?'}],
            'answer': '15',
            'expect_acc': 0.1,
        },
    ])

    transformed = apply_ppo_data_transform(dataset, _build_args(ppo_data_transform='opsd'), split='train')

    assert len(transformed) == 1
    assert transformed[0]['messages'] == [{'role': 'user', 'content': 'What is 5+5?'}]
    assert 'candidate answer that may be correct' in transformed[0]['teacher_prompt']
    assert '10' in transformed[0]['teacher_prompt']


def test_ppo_data_transform_opsd_with_label():
    dataset = HfDataset.from_list([
        {
            'messages': [{'role': 'user', 'content': 'What is 5+5?'}],
            'answer': '10',
            'ground_truth': '10',
            'expect_acc': 0.95,
        },
    ])

    transformed = apply_ppo_data_transform(
        dataset,
        _build_args(
            ppo_data_transform='opsd',
            ppo_data_label_key='ground_truth',
            ppo_data_include_label_in_teacher_prompt=True,
        ),
        split='train')

    assert len(transformed) == 1
    assert 'The ground-truth answer is' in transformed[0]['teacher_prompt']
    assert '10' in transformed[0]['teacher_prompt']
