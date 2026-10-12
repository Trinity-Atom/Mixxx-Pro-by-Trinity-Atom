#include <gtest/gtest.h>

#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QMessageBox>
#include <QStandardPaths>
#include <QTimer>

#include "library/stemgeneration.h"
#include "sources/soundsourceproxy.h"
#include "test/mixxxtest.h"
#include "track/track.h"

namespace {

class StemGenerationTest : public MixxxTest {
  protected:
    void SetUp() override {
        const QString python = QStandardPaths::findExecutable("python");
        if (python.isEmpty()) {
            GTEST_SKIP() << "Python is required for process integration tests";
        }
        const QDir root = getTestDataDir();
        ASSERT_TRUE(root.mkpath("tools"));
        ASSERT_TRUE(writeFile(root.filePath("model.onnx"), "test model"));
        ASSERT_TRUE(writeFile(root.filePath("model.json"), "{}"));
        ASSERT_TRUE(writeFile(root.filePath("tools/demucs_onnx_to_stem.py"),
                "import pathlib, sys, time\n"
                "args = sys.argv\n"
                "source = pathlib.Path(args[1])\n"
                "work = pathlib.Path(args[args.index('--temp-dir') + 1])\n"
                "(work / 'partial.f32').write_bytes(b'partial')\n"
                "print('STEM_PROGRESS 25', flush=True)\n"
                "if source.read_text() == 'slow':\n"
                "    cancel = pathlib.Path(args[args.index('--cancel-file') + 1])\n"
                "    while not cancel.exists(): time.sleep(0.05)\n"
                "    sys.exit(130)\n"
                "if source.read_text() == 'fail':\n"
                "    print('Model inference failed', file=sys.stderr)\n"
                "    sys.exit(2)\n"
                "pathlib.Path(args[args.index('--output') + 1]).write_bytes(b'stem')\n"));
        m_settings = UserSettingsPointer(new UserSettings(
                root.filePath("settings.cfg"), root.path(), root.path()));
        m_settings->setValue(ConfigKey("[StemGeneration]", "Python"), python);
        m_settings->setValue(ConfigKey("[StemGeneration]", "FFmpeg"), python);
        m_settings->setValue(ConfigKey("[StemGeneration]", "Model"), root.filePath("model.onnx"));
    }

    bool writeFile(const QString& path, const QByteArray& data) {
        QFile file(path);
        return file.open(QIODevice::WriteOnly) && file.write(data) == data.size();
    }

    void run(StemGeneration* generator, const QStringList& paths,
            bool cancel = false, int timeoutMs = 10000) {
        QEventLoop loop;
        QTimer timeout;
        timeout.setSingleShot(true);
        bool done = false;
        QObject::connect(generator, &StemGeneration::trackCreated,
                &loop, [this](const QString& path) { m_createdPaths.append(path); });
        QObject::connect(generator, &StemGeneration::completed, &loop,
                [&](int created, const QStringList& errors, bool canceled) {
                    m_created = created;
                    m_errors = errors;
                    m_canceled = canceled;
                    done = true;
                    loop.quit();
                });
        QObject::connect(&timeout, &QTimer::timeout, &loop, &QEventLoop::quit);
        generator->start(paths, nullptr);
        if (cancel) {
            QTimer::singleShot(200, generator, &StemGeneration::cancel);
        }
        if (!done) {
            timeout.start(timeoutMs);
            loop.exec();
        }
        EXPECT_TRUE(done) << "Stem generation did not finish";
        EXPECT_TRUE(getTestDataDir().entryList(
                {".mixxx-stems-*"}, QDir::Dirs | QDir::Hidden | QDir::NoDotAndDotDot).isEmpty());
    }

    UserSettingsPointer m_settings;
    QStringList m_createdPaths;
    QStringList m_errors;
    int m_created{0};
    bool m_canceled{false};
};

TEST(StemGenerationPathTest, KeepsOriginalDirectoryAndBaseName) {
    const QString source = QDir::temp().filePath("artist.track.flac");
    EXPECT_QSTRING_EQ(QDir::temp().filePath("artist.track.stem.mp4"),
            StemGeneration::outputPath(source));
    EXPECT_TRUE(StemGeneration::outputPath("track.STEM.MP4").isEmpty());
    EXPECT_TRUE(StemGeneration::outputPath("track.stem.m4a").isEmpty());
}

TEST_F(StemGenerationTest, CreatesQueueAndPreservesOriginals) {
    const QString first = getTestDataDir().filePath("one.flac");
    const QString second = getTestDataDir().filePath("two.mp3");
    ASSERT_TRUE(writeFile(first, "audio"));
    ASSERT_TRUE(writeFile(second, "audio"));
    StemGeneration generator(nullptr, m_settings);
    run(&generator, {first, second});
    EXPECT_EQ(m_created, 2);
    EXPECT_TRUE(m_errors.isEmpty());
    EXPECT_FALSE(m_canceled);
    EXPECT_EQ(m_createdPaths.size(), 2);
    EXPECT_TRUE(QFileInfo::exists(first));
    EXPECT_TRUE(QFileInfo::exists(second));
}

TEST_F(StemGenerationTest, DiscoversDependenciesWithoutSavedSettings) {
    if (QStandardPaths::findExecutable("ffmpeg").isEmpty()) {
        GTEST_SKIP() << "FFmpeg is required for automatic discovery tests";
    }
    ASSERT_TRUE(getTestDataDir().mkpath("models"));
    ASSERT_TRUE(QFile::copy(getTestDataDir().filePath("model.onnx"),
            getTestDataDir().filePath("models/htdemucs.onnx")));
    ASSERT_TRUE(QFile::copy(getTestDataDir().filePath("model.json"),
            getTestDataDir().filePath("models/htdemucs.json")));
    m_settings->remove(ConfigKey("[StemGeneration]", "Python"));
    m_settings->remove(ConfigKey("[StemGeneration]", "FFmpeg"));
    m_settings->remove(ConfigKey("[StemGeneration]", "Model"));
    const QString original = getTestDataDir().filePath("automatic.mp3");
    ASSERT_TRUE(writeFile(original, "audio"));
    StemGeneration generator(nullptr, m_settings);
    run(&generator, {original});
    EXPECT_EQ(m_created, 1);
    EXPECT_TRUE(m_errors.isEmpty());
}

TEST_F(StemGenerationTest, DiscoversDependenciesWhenSavedPathsAreStale) {
    if (QStandardPaths::findExecutable("ffmpeg").isEmpty()) {
        GTEST_SKIP() << "FFmpeg is required for automatic discovery tests";
    }
    ASSERT_TRUE(getTestDataDir().mkpath("models"));
    ASSERT_TRUE(QFile::copy(getTestDataDir().filePath("model.onnx"),
            getTestDataDir().filePath("models/htdemucs.onnx")));
    ASSERT_TRUE(QFile::copy(getTestDataDir().filePath("model.json"),
            getTestDataDir().filePath("models/htdemucs.json")));
    for (const auto& key : {"Python", "FFmpeg", "Model"}) {
        m_settings->setValue(ConfigKey("[StemGeneration]", key),
                getTestDataDir().filePath("missing-component"));
    }
    const QString original = getTestDataDir().filePath("stale-paths.mp3");
    ASSERT_TRUE(writeFile(original, "audio"));
    StemGeneration generator(nullptr, m_settings);
    run(&generator, {original});
    EXPECT_EQ(m_created, 1);
    EXPECT_TRUE(m_errors.isEmpty());
}

TEST_F(StemGenerationTest, MissingHelperReportsErrorWithoutConfigurationDialog) {
    ASSERT_TRUE(QFile::remove(getTestDataDir().filePath("tools/demucs_onnx_to_stem.py")));
    const QString original = getTestDataDir().filePath("unavailable.mp3");
    ASSERT_TRUE(writeFile(original, "audio"));
    bool reported = false;
    QTimer::singleShot(0, application(), [&] {
        for (auto* widget : QApplication::topLevelWidgets()) {
            auto* message = qobject_cast<QMessageBox*>(widget);
            if (message) {
                EXPECT_TRUE(message->text().contains("helper is missing"));
                reported = true;
                message->accept();
            }
        }
    });
    StemGeneration generator(nullptr, m_settings);
    generator.start({original}, nullptr);
    EXPECT_TRUE(reported);
    EXPECT_FALSE(QFileInfo::exists(StemGeneration::outputPath(original)));
}

TEST_F(StemGenerationTest, ReportsExistingOutputsMissingSourcesAndStemInputs) {
    const QString existing = getTestDataDir().filePath("existing.mp3");
    const QString destination = StemGeneration::outputPath(existing);
    ASSERT_TRUE(writeFile(existing, "audio"));
    ASSERT_TRUE(writeFile(destination, "keep original stem"));
    StemGeneration generator(nullptr, m_settings);
    run(&generator, {existing, destination, getTestDataDir().filePath("missing.mp3")});
    EXPECT_EQ(m_created, 0);
    EXPECT_EQ(m_errors.size(), 3);
    QFile file(destination);
    ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    EXPECT_EQ(file.readAll(), "keep original stem");
}

TEST_F(StemGenerationTest, ReportsInferenceFailureAndContinuesQueue) {
    const QString first = getTestDataDir().filePath("failed.mp3");
    const QString second = getTestDataDir().filePath("success.mp3");
    ASSERT_TRUE(writeFile(first, "fail"));
    ASSERT_TRUE(writeFile(second, "audio"));
    StemGeneration generator(nullptr, m_settings);
    run(&generator, {first, second});
    EXPECT_EQ(m_created, 1);
    ASSERT_EQ(m_errors.size(), 1);
    EXPECT_TRUE(m_errors.first().contains("Model inference failed"));
}

TEST_F(StemGenerationTest, RejectsCollidingOutputNames) {
    const QString first = getTestDataDir().filePath("same.mp3");
    const QString second = getTestDataDir().filePath("same.flac");
    ASSERT_TRUE(writeFile(first, "audio"));
    ASSERT_TRUE(writeFile(second, "audio"));
    StemGeneration generator(nullptr, m_settings);
    run(&generator, {first, second});
    EXPECT_EQ(m_created, 1);
    ASSERT_EQ(m_errors.size(), 1);
    EXPECT_TRUE(m_errors.first().contains("same output filename"));
}

TEST_F(StemGenerationTest, CancelStopsProcessAndRemovesTemporaryFiles) {
    const QString first = getTestDataDir().filePath("slow.mp3");
    const QString second = getTestDataDir().filePath("pending.mp3");
    ASSERT_TRUE(writeFile(first, "slow"));
    ASSERT_TRUE(writeFile(second, "audio"));
    StemGeneration generator(nullptr, m_settings);
    run(&generator, {first, second}, true);
    EXPECT_EQ(m_created, 0);
    EXPECT_TRUE(m_canceled);
    EXPECT_FALSE(QFileInfo::exists(StemGeneration::outputPath(first)));
    EXPECT_FALSE(QFileInfo::exists(StemGeneration::outputPath(second)));
}

TEST_F(StemGenerationTest, RealModelGeneratesReadableStem) {
    const QString model = qEnvironmentVariable("MIXXX_STEM_TEST_MODEL");
    const QString helper = qEnvironmentVariable("MIXXX_STEM_TEST_HELPER");
    const QString ffmpeg = qEnvironmentVariable("MIXXX_STEM_TEST_FFMPEG");
    if (model.isEmpty() || helper.isEmpty() || ffmpeg.isEmpty()) {
        GTEST_SKIP() << "Set MIXXX_STEM_TEST_MODEL, MIXXX_STEM_TEST_HELPER and "
                       "MIXXX_STEM_TEST_FFMPEG to run real-model integration";
    }
    const QString destination = getTestDataDir().filePath("tools/demucs_onnx_to_stem.py");
    ASSERT_TRUE(QFile::remove(destination));
    ASSERT_TRUE(QFile::copy(helper, destination));
    const QString original = getTestDataDir().filePath("real.wav");
    ASSERT_TRUE(QFile::copy(getTestDir().filePath("stems/stem01/sin_mainmix.wav"), original));
    m_settings->setValue(ConfigKey("[StemGeneration]", "Model"), model);
    m_settings->setValue(ConfigKey("[StemGeneration]", "FFmpeg"), ffmpeg);
    StemGeneration generator(nullptr, m_settings);
    run(&generator, {original}, false, 180000);
    ASSERT_EQ(m_created, 1);
    EXPECT_TRUE(m_errors.isEmpty());
    TrackPointer track(Track::newTemporary(StemGeneration::outputPath(original)));
    ASSERT_TRUE(SoundSourceProxy::isFileTypeSupported("stem.mp4") ||
            SoundSourceProxy::registerProviders());
    mixxx::AudioSource::OpenParams params;
    params.setChannelCount(mixxx::audio::ChannelCount::stereo());
    ASSERT_NE(SoundSourceProxy(track).openAudioSource(params), nullptr);
    EXPECT_EQ(track->getStemInfo().size(), 4);
}

TEST_F(StemGenerationTest, ShutdownStopsActiveJobAndCleansTemporaryFiles) {
    const QString original = getTestDataDir().filePath("shutdown.mp3");
    ASSERT_TRUE(writeFile(original, "slow"));
    StemGeneration generator(nullptr, m_settings);
    QEventLoop loop;
    generator.start({original}, nullptr);
    QTimer::singleShot(200, &loop, [&] {
        generator.stop();
        loop.quit();
    });
    loop.exec();
    EXPECT_FALSE(QFileInfo::exists(StemGeneration::outputPath(original)));
    EXPECT_TRUE(getTestDataDir().entryList(
            {".mixxx-stems-*"}, QDir::Dirs | QDir::Hidden | QDir::NoDotAndDotDot).isEmpty());
}

} // namespace
