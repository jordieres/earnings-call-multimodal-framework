import pandas as pd
from typing import Optional

from earningscall_framework.snalysis_response_patterns.config import PipelineConfig
from earningscall_framework.snalysis_response_patterns.io.transcript_loader import TranscriptQALoader
from earningscall_framework.snalysis_response_patterns.topics.topic_modeler import TopicModeler
from earningscall_framework.snalysis_response_patterns.topics.keyword_extractor import KeywordExtractor
from earningscall_framework.snalysis_response_patterns.topics.topic_labeler import TopicLabeler
from earningscall_framework.snalysis_response_patterns.features.emotion_feature_builder import EmotionFeatureBuilder
from earningscall_framework.snalysis_response_patterns.stats.effect_sizes import StatsTester
from earningscall_framework.snalysis_response_patterns.plotting.answer_plotter import AnswerPlotter

class CompanyPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        loader: TranscriptQALoader,
        topic_modeler: TopicModeler,
        kw_extractor: KeywordExtractor,
        topic_labeler: TopicLabeler,
        feature_builder: EmotionFeatureBuilder,
        stats: StatsTester,
        plotter: Optional[AnswerPlotter] = None,
    ):
        self.config = config
        self.loader = loader
        self.topic_modeler = topic_modeler
        self.kw_extractor = kw_extractor
        self.topic_labeler = topic_labeler
        self.feature_builder = feature_builder
        self.stats = stats
        self.plotter = plotter

    def run(self, company: str, plot_if_company: Optional[str] = None) -> pd.DataFrame:
        df = self.loader.load_company(company)
        df, topic_info = self.topic_modeler.add_topics(df)
        df = self.kw_extractor.add_keywords(df)
        df, _topic_labels = self.topic_labeler.add_topic_labels(topic_info, df)

        if self.plotter is not None and plot_if_company == company:
            self.plotter.plot_answers_by_topic(df)

        df_audio, df_text = self.feature_builder.build_audio_text_views(df)

        df_tests = self.stats.compute_tests(df_audio, df_text)
        df_tests = self.stats.add_hedges_g_ci(df_tests, "audio")
        df_tests = self.stats.add_hedges_g_ci(df_tests, "text")
        return df_tests
