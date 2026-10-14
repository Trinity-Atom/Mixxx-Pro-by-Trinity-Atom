#include "library/stemgeneration.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMessageBox>
#include <QProgressDialog>
#include <QSet>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTimer>

#include "moc_stemgeneration.cpp"
#include "preferences/configobject.h"
#include "util/logger.h"

namespace {
const mixxx::Logger kLogger("StemGeneration");
const ConfigKey kPythonKey("[StemGeneration]", "Python");
const ConfigKey kModelKey("[StemGeneration]", "Model");
const ConfigKey kFfmpegKey("[StemGeneration]", "FFmpeg");

QString resolveExecutable(const QString& value) {
    if (value.isEmpty()) {
        return {};
    }
    return QStandardPaths::findExecutable(value);
}

QString firstExecutable(const QStringList& candidates) {
    for (const auto& candidate : candidates) {
        const QString executable = resolveExecutable(candidate);
        if (!executable.isEmpty()) {
            return executable;
        }
    }
    return {};
}

QString firstModel(const QStringList& candidates) {
    for (const auto& candidate : candidates) {
        if (candidate.isEmpty()) {
            continue;
        }
        const QFileInfo model(candidate);
        const QString contract = model.dir().filePath(model.completeBaseName() + ".json");
        if (model.isFile() && QFileInfo(contract).isFile()) {
            return model.absoluteFilePath();
        }
    }
    return {};
}
} // namespace

StemGeneration::StemGeneration(QObject* parent, UserSettingsPointer config)
        : QObject(parent), m_config(std::move(config)) {
    connect(&m_process, &QProcess::readyReadStandardOutput, this, &StemGeneration::readOutput);
    connect(&m_process, &QProcess::readyReadStandardError, this, &StemGeneration::readOutput);
    connect(&m_process, &QProcess::finished, this, &StemGeneration::finishTrack);
    connect(&m_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        if (error == QProcess::FailedToStart && m_running && !m_stopping) {
            m_stderr += m_process.errorString().toUtf8();
            finishTrack(-1, QProcess::CrashExit);
        }
    });
}

StemGeneration::~StemGeneration() {
    stop();
}

QString StemGeneration::outputPath(const QString& inputPath) {
    const QFileInfo input(inputPath);
    if (input.fileName().endsWith(".stem.mp4", Qt::CaseInsensitive) ||
            input.fileName().endsWith(".stem.m4a", Qt::CaseInsensitive)) {
        return {};
    }
    return input.dir().filePath(input.completeBaseName() + QStringLiteral(".stem.mp4"));
}

QStringList StemGeneration::findDependencies() {
    const QDir resources(m_config->getResourcePath());
    const QDir settings(m_config->getSettingsPath());
#ifdef Q_OS_WIN
    const QString pythonName = QStringLiteral("python.exe");
    const QString ffmpegName = QStringLiteral("ffmpeg.exe");
#else
    const QString pythonName = QStringLiteral("python3");
    const QString ffmpegName = QStringLiteral("ffmpeg");
#endif
    m_python = firstExecutable({m_config->getValueString(kPythonKey),
            resources.filePath("tools/python/" + pythonName),
            pythonName, QStringLiteral("python")});
    m_ffmpeg = firstExecutable({m_config->getValueString(kFfmpegKey),
            resources.filePath("tools/" + ffmpegName),
            resources.filePath(ffmpegName), ffmpegName});
    m_model = firstModel({m_config->getValueString(kModelKey),
            resources.filePath("models/htdemucs.onnx"),
            settings.filePath("models/htdemucs.onnx"),
            QDir(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation))
                    .filePath("mixxx/htdemucs.onnx")});
    QStringList errors;
    if (m_python.isEmpty()) {
        errors.append(tr("The Python runtime required for Stem generation is missing."));
    }
    if (m_ffmpeg.isEmpty()) {
        errors.append(tr("The FFmpeg executable required for Stem generation is missing."));
    }
    if (m_model.isEmpty()) {
        errors.append(tr("The Stem separation model or its matching JSON contract is missing."));
    }
    if (helperPath().isEmpty()) {
        errors.append(tr("The Stem generation helper is missing."));
    }
    return errors;
}

QString StemGeneration::helperPath() const {
    const QDir resources(m_config->getResourcePath());
    const QString resourceHelper = resources.filePath("tools/demucs_onnx_to_stem.py");
    if (QFileInfo(resourceHelper).isFile()) {
        return resourceHelper;
    }
#ifdef MIXXX_STEM_SOURCE_DIR
    const QString sourceResources =
            QDir(QStringLiteral(MIXXX_STEM_SOURCE_DIR)).filePath("res");
    if (QDir::cleanPath(resources.absolutePath()) == QDir::cleanPath(sourceResources)) {
        const QString sourceHelper =
                QDir(QStringLiteral(MIXXX_STEM_SOURCE_DIR))
                        .filePath("tools/demucs_onnx_to_stem.py");
        if (QFileInfo(sourceHelper).isFile()) {
            return sourceHelper;
        }
    }
#endif
    return {};
}

void StemGeneration::start(const QStringList& paths, QWidget* parent) {
    if (m_running) {
        m_progress->show();
        m_progress->raise();
        QMessageBox::information(parent, tr("Create Stems"),
                tr("Stem generation is already running. Wait for it to finish "
                   "before starting another selection."));
        return;
    }
    if (paths.isEmpty()) {
        QMessageBox::warning(parent, tr("Create Stems"), tr("No tracks selected."));
        return;
    }
    const QStringList dependencyErrors = findDependencies();
    if (!dependencyErrors.isEmpty()) {
        for (const auto& error : dependencyErrors) {
            kLogger.warning() << error;
        }
        QMessageBox::warning(parent, tr("Create Stems"),
                tr("Stem generation is unavailable because required components "
                   "could not be found:\n\n%1").arg(dependencyErrors.join("\n")));
        return;
    }
    m_pending.clear();
    m_errors.clear();
    QSet<QString> destinations;
    for (const auto& path : paths) {
        const QString destination = outputPath(path);
        QString key = QFileInfo(destination).absoluteFilePath();
#ifdef Q_OS_WIN
        key = key.toCaseFolded();
#endif
        if (destination.isEmpty()) {
            m_errors.append(tr("%1: already a Stem file.").arg(path));
        } else if (!QFileInfo(path).isFile()) {
            m_errors.append(tr("%1: source file is missing.").arg(path));
        } else if (QFileInfo::exists(destination)) {
            m_errors.append(tr("%1: output already exists; it was not overwritten.").arg(destination));
        } else if (destinations.contains(key)) {
            m_errors.append(tr("%1: another selected track has the same output filename.").arg(path));
        } else {
            destinations.insert(key);
            m_pending.append(path);
        }
    }
    m_total = m_pending.size();
    m_processed = 0;
    m_created = 0;
    m_canceled = false;
    m_running = true;
    m_progress = std::make_unique<QProgressDialog>(
            tr("Preparing Stem generation..."), tr("Cancel"), 0, m_total * 100);
    m_progress->setWindowTitle(tr("Create Stems"));
    m_progress->setWindowModality(Qt::NonModal);
    m_progress->setAutoClose(false);
    m_progress->setAutoReset(false);
    connect(m_progress.get(), &QProgressDialog::canceled, this, &StemGeneration::cancel);
    m_progress->show();
    startNext();
}

void StemGeneration::startNext() {
    if (!m_running || m_stopping) {
        return;
    }
    if (m_pending.isEmpty() || m_canceled) {
        finish();
        return;
    }
    m_input = m_pending.takeFirst();
    m_output = outputPath(m_input);
    m_stdout.clear();
    m_stderr.clear();
    m_temporaryDir = std::make_unique<QTemporaryDir>(
            QFileInfo(m_output).dir().filePath(".mixxx-stems-XXXXXX"));
    if (!m_temporaryDir->isValid()) {
        m_stderr = tr("Cannot create a temporary directory next to the source file.").toUtf8();
        finishTrack(-1, QProcess::CrashExit);
        return;
    }
    m_progress->setLabelText(tr("Creating stems (%1 of %2): %3")
                                    .arg(m_processed + 1)
                                    .arg(m_total)
                                    .arg(QFileInfo(m_input).fileName()));
    m_progress->setValue(m_processed * 100);
    m_process.start(m_python,
            {helperPath(),
                    m_input, "--output", m_output,
                    "--model", m_model,
                    "--ffmpeg", m_ffmpeg,
                    "--threads", "4", "--progress",
                    "--temp-dir", m_temporaryDir->path(),
                    "--cancel-file", QDir(m_temporaryDir->path()).filePath("cancel")});
}

void StemGeneration::readOutput() {
    m_stderr += m_process.readAllStandardError();
    m_stderr = m_stderr.right(16384);
    m_stdout += m_process.readAllStandardOutput();
    int newline;
    while ((newline = m_stdout.indexOf('\n')) >= 0) {
        const QByteArray line = m_stdout.left(newline).trimmed();
        m_stdout.remove(0, newline + 1);
        if (line.startsWith("STEM_PROGRESS ")) {
            bool ok = false;
            const int percent = line.mid(14).toInt(&ok);
            if (ok && m_progress && !m_canceled) {
                m_progress->setValue(m_processed * 100 + qBound(0, percent, 99));
            }
        } else if (line.startsWith("STEM_BACKEND ")) {
            kLogger.info() << QString::fromUtf8(line);
        }
    }
    m_stdout = m_stdout.right(16384);
}

void StemGeneration::finishTrack(int exitCode, QProcess::ExitStatus status) {
    if (!m_running || m_stopping) {
        return;
    }
    readOutput();
    m_temporaryDir.reset();
    if (status == QProcess::NormalExit && exitCode == 0 && QFileInfo(m_output).isFile()) {
        ++m_created;
        emit trackCreated(m_output);
        if (!m_stderr.trimmed().isEmpty()) {
            m_errors.append(tr("%1: generated with warnings:\n%2")
                                    .arg(m_input, QString::fromUtf8(m_stderr)));
        }
    } else if (!m_canceled) {
        const QString details = m_stderr.trimmed().isEmpty()
                ? tr("The helper failed (exit code %1) or did not create the output.").arg(exitCode)
                : QString::fromUtf8(m_stderr);
        m_errors.append(tr("%1:\n%2").arg(m_input, details));
    }
    ++m_processed;
    QTimer::singleShot(0, this, &StemGeneration::startNext);
}

void StemGeneration::finish() {
    if (!m_running) {
        return;
    }
    m_running = false;
    m_progress->hide();
    for (const auto& error : m_errors) {
        kLogger.warning() << error;
    }
    emit completed(m_created, m_errors, m_canceled);
}

void StemGeneration::cancel() {
    if (!m_running) {
        return;
    }
    m_canceled = true;
    m_pending.clear();
    if (m_process.state() != QProcess::NotRunning) {
        m_progress->setLabelText(tr("Canceling Stem generation after the current segment..."));
        m_progress->setCancelButton(nullptr);
        m_progress->show();
        requestCancellation();
    } else {
        finish();
    }
}

void StemGeneration::requestCancellation() {
    QFile file(QDir(m_temporaryDir->path()).filePath("cancel"));
    if (!file.open(QIODevice::WriteOnly)) {
        kLogger.warning() << "Cannot request Stem generation cancellation:" << file.errorString();
        m_errors.append(tr("Cannot request cancellation: %1").arg(file.errorString()));
        m_process.kill();
    }
}

void StemGeneration::stop() {
    m_stopping = true;
    m_running = false;
    m_pending.clear();
    if (m_progress) {
        m_progress->hide();
    }
    if (m_process.state() != QProcess::NotRunning) {
        requestCancellation();
        if (!m_process.waitForFinished(30000)) {
            kLogger.warning() << "Stem generation did not stop within 30 seconds; terminating helper";
            m_process.kill();
            m_process.waitForFinished();
        }
    }
    m_temporaryDir.reset();
}
