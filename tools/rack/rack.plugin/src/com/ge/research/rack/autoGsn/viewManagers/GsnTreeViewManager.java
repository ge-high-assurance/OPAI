/*
 * BSD 3-Clause License
 * 
 * Copyright (c) 2023, General Electric Company and Galois, Inc.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.ge.research.rack.autoGsn.viewManagers;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

import org.eclipse.core.runtime.FileLocator;
import org.eclipse.core.runtime.Path;
import org.eclipse.core.runtime.Platform;
import org.osgi.framework.Bundle;

import java.net.URL;

/**
 * @author Saswata Paul Note: This class manages all the GsnTreeView for the AutoGsn feature.
 *     <p>When it is initially called by the JavafxAppLancgManager, This class creates a stage using
 *     GsnTreeView.fxml.
 */
public class GsnTreeViewManager extends Application {

    public static Stage stage;

    /**
     * This function creates a new stage and sets that as the class stage then it sets a provided
     * fxml to the class stage and returns the FXMLLoader object for future use if needed
     *
     * <p>It also sets the Title of the stage as a function of a passed currentGoalId
     *
     * @param fxmlFilePath
     * @return
     */
    public static FXMLLoader createNewStageAndSetFxml(String fxmlFilePath, String currentGoalId) {

        try {

            // Creating a new stage since this will be a second window
            // That will be displayed over an AutoGsnView Window
            stage = new Stage();

            Bundle bundle = Platform.getBundle("rack.plugin");
            URL fxmlUrl = FileLocator.find(bundle, new Path(fxmlFilePath), null);
            fxmlUrl = FileLocator.toFileURL(fxmlUrl);

            // Creating an FXMLLoader object that can be returned for use where needed
            FXMLLoader loader = new FXMLLoader(fxmlUrl);
            Parent root = loader.load();
            stage.setTitle("GSN Cascading View with " + currentGoalId + " in focus");
            stage.setScene(new Scene(root));
            // stage.setResizable(false); // User cannot change window size
            stage.show();

            // return the FXMLLoader
            return loader;

        } catch (Exception ex) {

            ex.printStackTrace();
        }

        System.out.println("Here");
        return null;
    }

    @Override
    public void start(Stage primaryStage) {

        try {

            // Nothing needs to be done here because in this case,
            // We want a new window to be opened along with the exsting window
            // So a new stage also needs to be created. This will be done when the caller
            // view's controller calls the createNewStageAndSetFxml() method

        } catch (Exception ex) {

            ex.printStackTrace();
        }
    }
}
